################################################################################
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import wraps
from typing import List, Tuple, Optional, Union
from math import log2, log
import os
import yaml
import json
import subprocess
import collections
from contextlib import contextmanager
import Tensile.TensileInstructions as ti
from Tensile.Common import detectGlobalCurrentISA, restoreDefaultGlobalParameters, \
    assignGlobalParameters, getGfxName, gfxArch, globalParameters

def kernel_header(name: str, gfx_arch: str, vgpr: int, sgpr: int, lds: int):
    vgpr = ((vgpr+7)//8)*8
    sgpr = ((sgpr+7)//8)*8
    lds  = ((lds+31)//32)*32

    return f'''
.amdgcn_target "amdgcn-amd-amdhsa--{gfx_arch}"
.text
.protected {name}
.globl {name}
.p2align 8
.type {name},@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel {name}
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset {vgpr} // accvgpr offset
  .amdhsa_next_free_vgpr {vgpr} // vgprs
  .amdhsa_next_free_sgpr {sgpr} // sgprs
  .amdhsa_group_segment_fixed_size {lds} // lds bytes
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
'''

@contextmanager
def asm_func(func_name: str, module: ti.Module):
    try:
        module.add(ti.TextBlock(f'{func_name}:\n'))
        yield
    finally:
        end_label_name = f'.L{func_name}_end'
        module.add(ti.SEndpgm())
        module.add(ti.TextBlock(f'{end_label_name}:\n'))
        module.add(ti.TextBlock(f'.size {func_name}, {end_label_name} - {func_name}\n'))

@contextmanager
def asm_loop(mod: ti.Module, name: str, it: str):
    try:
        loop_start_label = ti.Label(name, f'loop {name} starts')
        loop_end_label = ti.Label(f'{name}_end', f'loop {name} ends')
        mod.add(loop_start_label)
        mod.add(ti.SCmpEQU32(ti.sgpr(it), 0))
        mod.add(ti.SCBranchSCC1(loop_end_label.getLabelName()))
        mod.addSpaceLine()
        yield
    finally:
        mod.add(ti.SSubU32(ti.sgpr(it), ti.sgpr(it), 1))
        mod.add(ti.SBranch(loop_start_label.getLabelName()))
        mod.add(loop_end_label)
        mod.addSpaceLine()


class LayerNormKernelGenerator:
    srd_num_reg = 4
    srd_alignment = 4

    def __init__(self,
                 io_type: ti.DataType,
                 num_cols: int,
                 num_rows: int,
                 num_workitems: int,
                 arch: str):
        self.io_type = io_type
        self.num_workitems = num_workitems
        self.sgpr_pool = ti.RegisterPool(24, 's', True)
        self.vgpr_pool = ti.RegisterPool(40, 'v', True)
        self.sgpr_pool.add(0, 23) #TODO: estimate this
        self.vgpr_pool.add(0, 39) #TODO: estimate this
        self.debug_label = True
        self.arch = arch
        self.op = 'LayerNorm'
        self.sgprs  = collections.OrderedDict()
        self.vgprs  = collections.OrderedDict()

    @property
    def lds_usage_byte(self) -> int:
        return 32

    @property
    def func_name(self):
        return f'LayerNorm'

    def dumps(self, format: str) -> str:
        param_dict = {
            'io_type': self.io_type.toChar(),
            'num_workitems': self.num_workitems,
            'func_name': self.func_name,
            'debug_label': self.debug_label,
            'arch': self.arch,
            'op': self.op
        }

        if format.lower() == 'yaml':
            return yaml.dump(param_dict)
        elif format.lower() == 'json':
            return json.dumps(param_dict)
        else:
            assert False, f'Unsupported format {format}'

    def dump(self, format: str, output_path: str):
        s = self.dumps(format)
        with open(output_path, 'w') as f:
            f.write(s)

    def defineSgpr(self, name, numSgprs, align=1):
        if numSgprs == 0: return
        sgprIdx = self.sgpr_pool.checkOutAligned(numSgprs, align, tag=name, preventOverflow=0)
        self.sgprs[name] = sgprIdx
        return sgprIdx

    def defineVgpr(self, name, numVgprs, align=1):
        if numVgprs == 0: return
        vgprIdx = self.vgpr_pool.checkOutAligned(numVgprs, align, tag=name, preventOverflow=0)
        self.vgprs[name] = vgprIdx
        return vgprIdx


    def kernel_args(self):
        return (KernelArgument(8, 0, 'global_buffer', 'global'),
                KernelArgument(8, 8, 'global_buffer', 'global'),
                KernelArgument(4, 16, 'by_value'),
                KernelArgument(4, 20, 'by_value'),
                KernelArgument(4, 24, 'by_value'))


    def defineVariables(self):
        self.defineVgpr("Serial", 1, 1)
        self.defineVgpr("Src",    2, 2)
        self.defineVgpr("Dst",    2, 2)
        self.defineVgpr("Count",  1, 1)
        self.defineVgpr("Mean",   1, 1)
        self.defineVgpr("Std",    1, 1)
        self.defineVgpr("CountA", 1, 1)
        self.defineVgpr("MeanA",  1, 1)
        self.defineVgpr("StdA",   1, 1)
        self.defineVgpr("CountB", 1, 1)
        self.defineVgpr("MeanB",  1, 1)
        self.defineVgpr("StdB",   1, 1)
        self.defineVgpr("Widx",   1, 1)
        self.defineVgpr("Offset", 1, 1)
        self.defineVgpr("Value",  4, 4)
        self.defineVgpr("Tmp",    4, 1)

        self.defineSgpr("KernelArg", 2)
        self.defineSgpr("WorkGroup0", 1)
        self.defineSgpr("WorkGroup1", 1)
        self.defineSgpr("WorkGroup2", 1)
        self.defineSgpr("AddressIn", 2, 2)
        self.defineSgpr("AddressOut", 2, 2)
        self.defineSgpr("SizeLength", 1)
        self.defineSgpr("Eps", 1)
        self.defineSgpr("MainLoop", 1)
        self.defineSgpr("Offset", 1)
        self.defineSgpr("Src", 4, 4)
        self.defineSgpr("Dst", 4, 4)
        self.defineSgpr("Tmp", 6, 2)

        mod = ti.Module("defineVariables")

        for vkey in self.vgprs:
            mod.add(ti.RegSet("v", "vgpr"+vkey, self.vgprs[vkey]))
        mod.addSpaceLine()

        for skey in self.sgprs:
            mod.add(ti.RegSet("s", "sgpr"+skey, self.sgprs[skey]))
        mod.addSpaceLine()

        mod.add(ti.ValueSet("Srd127_96", "0x00020000", format=-1))
        mod.add(ti.ValueSet("BufferLimit", "0xffffffff", format=-1))
        mod.addSpaceLine()

        return mod


    def load_kernel_args(self):
        mod = ti.Module('Load kernel args')
        mod.addComment0('Load kernel args')
        mod.add(ti.SLoadB64(ti.sgpr("AddressIn", 2), ti.sgpr("KernelArg", 2), 0))
        mod.add(ti.SLoadB64(ti.sgpr("AddressOut", 2), ti.sgpr("KernelArg", 2), 8))
        mod.add(ti.SLoadB32(ti.sgpr("SizeLength"), ti.sgpr("KernelArg", 2), 20))
        mod.add(ti.SLoadB32(ti.sgpr("Eps"), ti.sgpr("KernelArg", 2), 24))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.addSpaceLine()
        return mod


    def init_param(self) -> ti.Module:
        mod = ti.Module("defineVariables")
        mod.addComment0("defineVariables")
        mod.add(ti.SMovB32(ti.sgpr("Src+0"), ti.sgpr("AddressIn+0")))
        mod.add(ti.SMovB32(ti.sgpr("Src+1"), ti.sgpr("AddressIn+1")))
        mod.add(ti.SMovB32(ti.sgpr("Src+2"), "BufferLimit"))
        mod.add(ti.SMovB32(ti.sgpr("Src+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ti.SMovB32(ti.sgpr("Dst+0"), ti.sgpr("AddressOut+0")))
        mod.add(ti.SMovB32(ti.sgpr("Dst+1"), ti.sgpr("AddressOut+1")))
        mod.add(ti.SMovB32(ti.sgpr("Dst+2"), "BufferLimit"))
        mod.add(ti.SMovB32(ti.sgpr("Dst+3"), "Srd127_96"))
        mod.addSpaceLine()

        mod.add(ti.VMovB32(ti.vgpr("Count"), 0.0))
        mod.add(ti.VMovB32(ti.vgpr("Mean"), 0.0))
        mod.add(ti.VMovB32(ti.vgpr("Std"), 0.0))
        mod.addSpaceLine()
        return mod


    def calculate_global_address(self) -> ti.Module:
        mod = ti.Module("calculate_global_address")
        mod.addComment0("calculate_global_address")
        mod.add(ti.SMulI32(ti.sgpr("Tmp"), ti.sgpr("WorkGroup1"), ti.sgpr("SizeLength")))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Offset"), "0x2", ti.vgpr("Serial")))
        mod.add(ti.VAddLShiftLeftU32(ti.vgpr("Offset"), "0x2", ti.sgpr("Tmp"), ti.vgpr("Offset")))
        mod.addSpaceLine()
        return mod


    def sum_per_data(self, val) -> ti.Module:
        mod = ti.Module("sum_per_data")
        mod.add(ti.VAddF32(ti.vgpr("Count"), ti.vgpr("Count"), 1.0))
        mod.add(ti.VSubF32(ti.vgpr("Tmp"), val, ti.vgpr("Mean")))  # delta
        mod.add(ti.VRcpF32(ti.vgpr("Tmp+1"), ti.vgpr("Count"))) # 1 / count
        mod.add(ti.VMulF32(ti.vgpr("Tmp+1"), ti.vgpr("Tmp"), ti.vgpr("Tmp+1"))) # delta / count
        mod.add(ti.VAddF32(ti.vgpr("Mean"), ti.vgpr("Mean"), ti.vgpr("Tmp+1"))) # new mean
        mod.add(ti.VSubF32(ti.vgpr("Tmp+1"), val, ti.vgpr("Mean")))
        mod.add(ti.VMulF32(ti.vgpr("Tmp"), ti.vgpr("Tmp"), ti.vgpr("Tmp+1")))
        mod.add(ti.VAddF32(ti.vgpr("Std"), ti.vgpr("Std"), ti.vgpr("Tmp")))
        mod.addSpaceLine()
        return mod


    def sum_per_threadx4(self) -> ti.Module:
        mod = ti.Module("sum_per_threadx4")
        mod.addComment0("sum_per_threadx4")
        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), 10, ti.sgpr("SizeLength")))
        with asm_loop(mod, "sum_per_threadx4", "MainLoop"):
            mod.add(ti.BufferLoadB128(ti.vgpr("Value",4), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.addSpaceLine()
            mod.add(self.sum_per_data(ti.vgpr("Value+0")))
            mod.add(self.sum_per_data(ti.vgpr("Value+1")))
            mod.add(self.sum_per_data(ti.vgpr("Value+2")))
            mod.add(self.sum_per_data(ti.vgpr("Value+3")))
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), 4096))
            mod.add(ti.VAddU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.sgpr("Tmp")))
            mod.addSpaceLine()
        return mod


    def adjusst_global_address(self) -> ti.Module:
        mod = ti.Module("adjusst_global_address")
        mod.add(ti.VMulLOU32(ti.vgpr("Tmp"), 3, ti.vgpr("Serial")))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), 2, ti.vgpr("Tmp")))
        mod.add(ti.VSubU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.vgpr("Tmp")))
        mod.addSpaceLine()
        return mod


    def sum_per_thread(self) -> ti.Module:
        mod = ti.Module("sum_per_thread")
        mod.addComment0("sum_per_thread")
        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), 8, ti.sgpr("SizeLength")))
        mod.add(ti.SAndB32(ti.sgpr("MainLoop"), ti.sgpr("MainLoop"), 0x3))
        with asm_loop(mod, "sum_per_thread", "MainLoop"):
            mod.add(ti.BufferLoadB32(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.addSpaceLine()
            mod.add(self.sum_per_data(ti.vgpr("Value")))
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), 1024))
            mod.add(ti.VAddU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.sgpr("Tmp")))
            mod.addSpaceLine()
        return mod


    def sum_in_some_thread(self)  -> ti.Module:
        label_sum_end = ti.Label("sum", f'loop sum end')
        mod = ti.Module("sum_in_some_thread")
        mod.addComment0("sum_in_some_thread")
        mod.add(ti.SAndB32(ti.sgpr("MainLoop"), ti.sgpr("SizeLength"), 255))
        mod.add(ti.VCmpLtU32("vcc", ti.vgpr("Serial"), ti.sgpr("MainLoop")))
        mod.add(ti.SCBranchVCCZ(label_sum_end.getLabelName()))
        mod.add(ti.SMovB64("exec", "vcc"))
        mod.add(ti.SNop(1))
        mod.add(ti.BufferLoadB32(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
        mod.add(ti.SWaitCnt(vmcnt=0))
        mod.addSpaceLine()
        mod.add(self.sum_per_data(ti.vgpr("Value")))
        mod.add(ti.SMovB64("exec", "-1"))
        mod.add(ti.SNop(1))
        mod.add(label_sum_end)
        mod.addSpaceLine()
        return mod


    def merge_sum(self) -> ti.Module:
        mod = ti.Module("merge_sum")
        mod.add(ti.VMovB32(ti.vgpr("CountA"), ti.vgpr("Count")))
        mod.add(ti.VMovB32(ti.vgpr("MeanA"), ti.vgpr("Mean")))
        mod.add(ti.VMovB32(ti.vgpr("StdA"), ti.vgpr("Std")))

        mod.add(ti.VSubF32(ti.vgpr("Tmp"), ti.vgpr("MeanB"), ti.vgpr("MeanA")))
        mod.add(ti.VAddF32(ti.vgpr("Count"), ti.vgpr("CountA"), ti.vgpr("CountB")))
        mod.add(ti.VRcpF32(ti.vgpr("Tmp+3"), ti.vgpr("Count")))
        mod.add(ti.VMulF32(ti.vgpr("Tmp+1"), ti.vgpr("CountA"), ti.vgpr("Tmp+3")))
        mod.add(ti.VMulF32(ti.vgpr("Tmp+2"), ti.vgpr("CountB"), ti.vgpr("Tmp+3")))
        mod.add(ti.VMulF32(ti.vgpr("MeanA"), ti.vgpr("MeanA"), ti.vgpr("Tmp+1")))
        mod.add(ti.VMulF32(ti.vgpr("MeanB"), ti.vgpr("MeanB"), ti.vgpr("Tmp+2")))
        mod.add(ti.VAddF32(ti.vgpr("Mean"), ti.vgpr("MeanA"), ti.vgpr("MeanB")))

        mod.add(ti.VAddF32(ti.vgpr("Std"), ti.vgpr("StdA"), ti.vgpr("StdB")))
        mod.add(ti.VMulF32(ti.vgpr("Tmp"), ti.vgpr("Tmp"), ti.vgpr("Tmp")))
        mod.add(ti.VMulF32(ti.vgpr("Tmp"), ti.vgpr("Tmp"), ti.vgpr("Tmp+1")))
        mod.add(ti.VMulF32(ti.vgpr("Tmp"), ti.vgpr("Tmp"), ti.vgpr("Tmp+2")))
        mod.add(ti.VMulF32(ti.vgpr("Tmp"), ti.vgpr("Tmp"), ti.vgpr("Count")))
        mod.add(ti.VAddF32(ti.vgpr("Std"), ti.vgpr("Std"), ti.vgpr("Tmp")))
        return mod


    def intra_wave_reduction(self) -> ti.Module:
        label = ti.Label("permute", f'permuge')
        mod = ti.Module("intra_wave_reduction")
        mod.addComment0("intra_wave_reduction")
        mod.add(ti.SMovB32(ti.sgpr("Tmp"), 1))
        mod.add(label)
        mod.addSpaceLine()
        mod.add(ti.VAddU32(ti.vgpr("Tmp"), ti.sgpr("Tmp"), ti.vgpr("Serial")))
        mod.add(ti.VAndB32(ti.vgpr("Tmp"), 63, ti.vgpr("Tmp")))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), 0x2, ti.vgpr("Tmp")))
        mod.addSpaceLine()
        mod.add(ti.DSBPermuteB32(ti.vgpr("CountB"), ti.vgpr("Tmp"), ti.vgpr("Count")))
        mod.add(ti.DSBPermuteB32(ti.vgpr("MeanB"), ti.vgpr("Tmp"), ti.vgpr("Mean")))
        mod.add(ti.DSBPermuteB32(ti.vgpr("StdB"), ti.vgpr("Tmp"), ti.vgpr("Std")))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.addSpaceLine()
        mod.add(self.merge_sum())
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), 1, ti.sgpr("Tmp")))
        mod.add(ti.SCmpLtU32(ti.sgpr("Tmp"), 64))
        mod.add(ti.SCBranchSCC1(label.getLabelName()))
        mod.addSpaceLine()
        return mod


    def inter_wave_reduction(self) -> ti.Module:
        label_inter = ti.Label("inter", f'inter')
        label_upper = ti.Label("upper", f'upper')
        label_lower = ti.Label("lower", f'lower')
        label_empty = ti.Label("empty", f'empty')
        label_end   = ti.Label("end", f'end')
        mod = ti.Module("inter_wave_reduction")
        mod.addComment0("inter_wave_reduction")
        mod.add(ti.VLShiftRightB32(ti.vgpr("Widx"), 6, ti.vgpr("Serial")))
        mod.add(ti.SMovB32(ti.sgpr("Offset"), 4))
        mod.add(label_inter)
        mod.add(ti.SLShiftRightB32(ti.sgpr("Offset"), 1, ti.sgpr("Offset")))
        mod.add(ti.SCmpEQU32(ti.sgpr("Offset"), 0))
        mod.add(ti.SCBranchSCC1(label_end.getLabelName()))
        mod.add(ti.SLShiftLeftB32(ti.sgpr("Tmp"), 1, ti.sgpr("Offset")))
        mod.add(ti.VCmpLtU32(ti.sgpr("Tmp+2",2), ti.vgpr("Widx"), ti.sgpr("Tmp")))
        mod.add(ti.VCmpGEU32(ti.sgpr("Tmp+4",2), ti.vgpr("Widx"), ti.sgpr("Offset")))
        mod.add(ti.SAndB64("vcc", ti.sgpr("Tmp+2",2), ti.sgpr("Tmp+4",2)))
        mod.add(ti.SCBranchVCCNZ(label_upper.getLabelName()))
        mod.add(ti.VCmpLtU32("vcc", ti.vgpr("Widx"), ti.sgpr("Offset")))
        mod.add(ti.SCBranchVCCNZ(label_lower.getLabelName()))
        mod.add(ti.SBranch(label_empty.getLabelName()))

        mod.add(label_upper)
        mod.add(ti.VSubU32(ti.vgpr("Tmp"), ti.vgpr("Widx"), ti.sgpr("Offset")))
        mod.add(ti.VMulLOU32(ti.vgpr("Tmp"), ti.vgpr("Tmp"), 4))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), 2, ti.vgpr("Tmp")))
        ds = ti.DSModifiers(offset=0)
        mod.add(ti.DSStoreB32(ti.vgpr("Tmp"), ti.vgpr("Count"), ds))
        ds = ti.DSModifiers(offset=4)
        mod.add(ti.DSStoreB32(ti.vgpr("Tmp"), ti.vgpr("Mean"), ds))
        ds = ti.DSModifiers(offset=8)
        mod.add(ti.DSStoreB32(ti.vgpr("Tmp"), ti.vgpr("Std"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(ti.SBarrier())
        mod.add(ti.SBranch(label_inter.getLabelName()))
        mod.add(label_lower)
        mod.add(ti.SBarrier())
        mod.add(ti.VMulLOU32(ti.vgpr("Tmp"), ti.vgpr("Widx"), 4))
        mod.add(ti.VLShiftLeftB32(ti.vgpr("Tmp"), 2, ti.vgpr("Tmp")))
        ds = ti.DSModifiers(offset=0)
        mod.add(ti.DSLoadB32(ti.vgpr("CountB"), ti.vgpr("Tmp"), ds))
        ds = ti.DSModifiers(offset=4)
        mod.add(ti.DSLoadB32(ti.vgpr("MeanB"), ti.vgpr("Tmp"), ds))
        ds = ti.DSModifiers(offset=8)
        mod.add(ti.DSLoadB32(ti.vgpr("StdB"), ti.vgpr("Tmp"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(self.merge_sum())
        mod.add(ti.SBranch(label_inter.getLabelName()))
        mod.add(label_empty)
        mod.add(ti.SBarrier())
        mod.add(ti.SBranch(label_inter.getLabelName()))
        mod.add(label_end)
        mod.addSpaceLine()
        return mod


    def broadcast(self) -> ti.Module:
        label_lower = ti.Label("broadcast_lower", f'broadcast_lower')
        label_end = ti.Label("broadcast_end", f'broadcast_end')

        mod = ti.Module("broadcast")
        mod.addComment0("broadcast")
        mod.add(ti.VCmpEQU32("vcc", ti.vgpr("Widx"), 0))
        mod.add(ti.SCBranchVCCZ(label_lower.getLabelName()))
        ds = ti.DSModifiers(offset=0)
        mod.add(ti.DSStoreB32(ti.vgpr("Widx"), ti.vgpr("Count"), ds))
        ds = ti.DSModifiers(offset=4)
        mod.add(ti.DSStoreB32(ti.vgpr("Widx"), ti.vgpr("Mean"), ds))
        ds = ti.DSModifiers(offset=8)
        mod.add(ti.DSStoreB32(ti.vgpr("Widx"), ti.vgpr("Std"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(ti.SBarrier())
        mod.add(ti.SBranch(label_end.getLabelName()))
        mod.add(label_lower)
        mod.add(ti.SBarrier())
        mod.add(ti.VMovB32(ti.vgpr("Tmp"), 0))
        ds = ti.DSModifiers(offset=0)
        mod.add(ti.DSLoadB32(ti.vgpr("Count"), ti.vgpr("Tmp"), ds))
        ds = ti.DSModifiers(offset=4)
        mod.add(ti.DSLoadB32(ti.vgpr("Mean"), ti.vgpr("Tmp"), ds))
        ds = ti.DSModifiers(offset=8)
        mod.add(ti.DSLoadB32(ti.vgpr("Std"), ti.vgpr("Tmp"), ds))
        mod.add(ti.SWaitCnt(lgkmcnt=0))
        mod.add(label_end)
        mod.addSpaceLine()
        return mod


    def get_average(self) -> ti.Module:
        mod = ti.Module("get_average")
        mod.addComment0("get_average")
        mod.add(ti.VCvtI32toF32(ti.vgpr("Tmp"), ti.sgpr("SizeLength")))
        mod.add(ti.VRcpF32(ti.vgpr("Tmp"),ti.vgpr("Tmp")))
        mod.add(ti.VMulF32(ti.vgpr("Std"), ti.vgpr("Tmp"), ti.vgpr("Std")))

        mod.add(ti.VAddF32(ti.vgpr("Std"), ti.vgpr("Std"), ti.sgpr("Eps")))
        mod.add(ti.TextBlock("v_rsq_f32 v[vgprStd], v[vgprStd]\n"))
        mod.addSpaceLine()
        return mod


    def layernorm_cal(self, val) -> ti.Module:
        mod = ti.Module("layernorm_cal")
        mod.add(ti.VSubF32(val, val, ti.vgpr("Mean")))
        mod.add(ti.VMulF32(val, val, ti.vgpr("Std")))
        return mod


    def layernorm_threadx4(self) -> ti.Module:
        mod = ti.Module("layernorm_threadx4")
        mod.addComment0("layernorm_threadx4")
        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), 10, ti.sgpr("SizeLength")))
        with asm_loop(mod, "layernorm_threadx4", "MainLoop"):
            mod.add(ti.BufferLoadB128(ti.vgpr("Value",4), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.add(self.layernorm_cal(ti.vgpr("Value+0")))
            mod.add(self.layernorm_cal(ti.vgpr("Value+1")))
            mod.add(self.layernorm_cal(ti.vgpr("Value+2")))
            mod.add(self.layernorm_cal(ti.vgpr("Value+3")))
            mod.add(ti.BufferStoreB128(ti.vgpr("Value",4), ti.vgpr("Offset"), ti.sgpr("Dst",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), 4096))
            mod.add(ti.VAddU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.sgpr("Tmp")))
            mod.addSpaceLine()
        return mod


    def layernorm_thread(self) -> ti.Module:
        mod = ti.Module("layernorm_thread")
        mod.addComment0("layernorm_thread")
        mod.add(ti.SLShiftRightB32(ti.sgpr("MainLoop"), 8, ti.sgpr("SizeLength")))
        mod.add(ti.SAndB32(ti.sgpr("MainLoop"), ti.sgpr("MainLoop"), 0x3))
        with asm_loop(mod, "layernorm_thread", "MainLoop"):
            mod.add(ti.BufferLoadB32(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.add(self.layernorm_cal(ti.vgpr("Value")))
            mod.add(ti.BufferStoreB32(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Dst",4), 0, ti.MUBUFModifiers(offen=True)))
            mod.add(ti.SWaitCnt(vmcnt=0))
            mod.add(ti.SMovB32(ti.sgpr("Tmp"), 1024))
            mod.add(ti.VAddU32(ti.vgpr("Offset"), ti.vgpr("Offset"), ti.sgpr("Tmp")))
            mod.addSpaceLine()
        return mod


    def layernorm_in_some_thread(self)  -> ti.Module:
        label_layernorm_end = ti.Label("layernorm", f'loop layernorm end')
        mod = ti.Module("layernorm_in_some_thread")
        mod.addComment0("layernorm_in_some_thread")
        mod.add(ti.SAndB32(ti.sgpr("MainLoop"), ti.sgpr("SizeLength"), 255))
        mod.add(ti.VCmpLtU32("vcc", ti.vgpr("Serial"), ti.sgpr("MainLoop")))
        mod.add(ti.SCBranchVCCZ(label_layernorm_end.getLabelName()))
        mod.add(ti.SMovB64("exec", "vcc"))
        mod.add(ti.SNop(1))
        mod.add(ti.BufferLoadB32(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Src",4), 0, ti.MUBUFModifiers(offen=True)))
        mod.add(ti.SWaitCnt(vmcnt=0))
        mod.add(self.layernorm_cal(ti.vgpr("Value")))
        mod.add(ti.BufferStoreB32(ti.vgpr("Value"), ti.vgpr("Offset"), ti.sgpr("Dst",4), 0, ti.MUBUFModifiers(offen=True)))
        mod.add(ti.SWaitCnt(vmcnt=0))
        mod.add(ti.SMovB64("exec", "-1"))
        mod.add(ti.SNop(1))
        mod.add(label_layernorm_end)
        mod.addSpaceLine()
        return mod


    def layernorm_kernel_body(self) -> ti.Module:
        mod = ti.Module(self.func_name)
        mod.add(self.defineVariables())
        with asm_func(self.func_name, mod):
            mod.add(self.load_kernel_args())
            mod.add(self.init_param())
            mod.add(self.calculate_global_address())
            mod.add(self.sum_per_threadx4())
            mod.addSpaceLine()
            mod.add(self.adjusst_global_address())
            mod.add(self.sum_per_thread())
            mod.add(self.sum_in_some_thread())
            mod.add(self.intra_wave_reduction())
            mod.add(self.inter_wave_reduction())
            mod.add(self.broadcast())
            mod.add(self.get_average())
            mod.add(self.calculate_global_address())
            mod.add(self.layernorm_threadx4())
            mod.add(self.adjusst_global_address())
            mod.add(self.layernorm_thread())
            mod.add(self.layernorm_in_some_thread())
        return mod

def kernel_rodata(name: str):
    return f'''
.rodata
.p2align 6
.amdhsa_kernel {name}
.amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_accum_offset 8
.amdhsa_next_free_vgpr .amdgcn.next_free_vgpr
.amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
.end_amdhsa_kernel
'''

@dataclass
class KernelArgument:
    size: int
    offset: int
    value_kind: str
    address_space: Optional[str] = None

    def to_dict(self):
        d = {'.size': self.size, '.offset': self.offset,
             '.value_kind': self.value_kind}

        if self.address_space:
            d['.address_space'] = self.address_space

        return d

@dataclass
class KernelMeta:
    name: str
    num_vgpr: int
    num_sgpr: int
    num_agpr: int
    num_lds_bytes: int
    wavefront_size: int
    max_workgroup_size: int
    args_alignment: int
    args: List[KernelArgument]

    def update_args_offsets(self):
        offset = 0
        for arg in args:
            arg.offset = offset
            offset += arg.size

    def _get_args_size(self):
        total_size = sum(arg.size for arg in self.args)
        total_size += (self.args_alignment - (total_size % self.args_alignment))
        return total_size

    def to_dict(self):
        return {
            '.name': self.name,
            '.symbol': f'{self.name}.kd',
            '.kernarg_segment_size': self._get_args_size(),
            '.group_segment_fixed_size': self.num_lds_bytes,
            '.private_segment_fixed_size': 0,
            '.kernarg_segment_align': self.args_alignment,
            '.wavefront_size': self.wavefront_size,
            '.sgpr_count': self.num_sgpr,
            '.vgpr_count': self.num_vgpr,
            '.agpr_count': self.num_agpr,
            '.max_flat_workgroup_size': self.max_workgroup_size,
            '.args': [arg.to_dict() for arg in self.args]
        }

def meta_str(kernels: Tuple[KernelMeta]):
    beg = '.amdgpu_metadata\n---'
    content_str = yaml.dump({'amdhsa.version': [1, 1], 'amdhsa.kernels': [kernel.to_dict() for kernel in kernels]})
    end = '.end_amdgpu_metadata'
    return '\n'.join([beg, content_str, end, ''])


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-o', '--output', type=str, required=True, help='Output path of compiled binary')
    ap.add_argument('-m', type=int, default=16, help='Dimension 0 of tile')
    ap.add_argument('-n', type=int, default=16, help='Dimension 1 of tile')
    ap.add_argument('--toolchain', type=str, default='/opt/rocm/llvm/bin/clang++', help='Path to ROCm compiler')
    ap.add_argument('--debug-build', action='store_true', dest='debug_build', help='Build with debug information')
    ap.set_defaults(debug_build=False)
    ap.add_argument('--arch', type=str, default='gfx90a', help='Target architecture for assembler, e.g. gfx908. Default is gfx90a')
    args = ap.parse_args()
    output_path: str = args.output
    m: int = args.m
    n: int = args.n
    toolchain_path: str = args.toolchain
    debug_build: bool = args.debug_build
    arch: str = args.arch
    isa = gfxArch(arch)

    if any([not i for i in (arch, toolchain_path, isa)]):
        restoreDefaultGlobalParameters()
        assignGlobalParameters({})
        detectGlobalCurrentISA()
        isa = globalParameters['CurrentISA']
        arch = getGfxName(isa)
        toolchain_path = globalParameters['AssemblerPath']

    ti.Base._global_ti.init(isa, toolchain_path, False)
    layernorm = LayerNormKernelGenerator(ti.DataType('S'), n, m, 256, arch)
    kernel_body = layernorm.layernorm_kernel_body()
    args = layernorm.kernel_args()
    func_name = layernorm.func_name
    meta = KernelMeta(func_name, layernorm.vgpr_pool.size(), layernorm.sgpr_pool.size(), 0, layernorm.lds_usage_byte, 64, 256, 8, args)
    meta.update_args_offsets()
    k_str = '\n'.join([kernel_header(func_name, arch, layernorm.vgpr_pool.size(), layernorm.sgpr_pool.size(), layernorm.lds_usage_byte),
                       meta_str((meta,)),
                       str(kernel_body)])

    with open(output_path, 'w') as f:
        f.write(k_str)

    output_path_basename = os.path.splitext(output_path)[0]

    if debug_build:
        build_args = ['-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcode-object-version=4', f'-mcpu={arch}', '-mwavefrontsize64', '-c', '-g', '-o', f'{output_path_basename}.o', f'{output_path_basename}.s']
    else:
        build_args = ['-x', 'assembler', '-target', 'amdgcn-amd-amdhsa', '-mcode-object-version=4', f'-mcpu={arch}', '-mwavefrontsize64', '-c', '-o', f'{output_path_basename}.o', f'{output_path_basename}.s']

    ret = subprocess.run([toolchain_path] + build_args)
    ret = subprocess.run([toolchain_path, '-target', 'amdcgn-amdhsa', '-o', f'{output_path_basename}.co', f'{output_path_basename}.o'])
    layernorm.dump('yaml', f'{output_path_basename}.yaml')
