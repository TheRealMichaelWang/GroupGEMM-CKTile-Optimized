import re, sys
# usage: setcfg.py MT NT KT MW NW MWT NWT KWT PIPE(V3|V4|MEM|ASYNC|EW|PS) [SCHED] [BlocksPerCU] [Persistent true|false] [Epilogue cs|def]
a = sys.argv[1:]
MT,NT,KT,MW,NW,MWT,NWT,KWT = a[:8]; pipe = a[8]; sched = a[9] if len(a)>9 else "Intrawave"; bpc = a[10] if len(a)>10 else "1"; pers = a[11] if len(a)>11 else "true"; epi = a[12] if len(a)>12 else "cs"
f = "ck_kernel/ck_grouped_gemm_config.hpp"
s = open(f).read()
for k,v in [("M_Tile",MT),("N_Tile",NT),("K_Tile",KT),("M_Warp",MW),("N_Warp",NW),("M_Warp_Tile",MWT),("N_Warp_Tile",NWT),("K_Warp_Tile",KWT)]:
    s = re.sub(rf"(index_t {k} *= *)\d+;", rf"\g<1>{v};", s)
pmap = {k:"ck_tile::"+v for k,v in {"V3":"GemmPipelineAgBgCrCompV3","V4":"GemmPipelineAgBgCrCompV4","MEM":"GemmPipelineAgBgCrMem","ASYNC":"GemmPipelineAgBgCrCompAsync","EW":"GemmPipelineAgBgCrCompAsyncEightWaves","PS":"WeightPreshufflePipelineAGmemBGmemCRegV2"}.items()}
pmap["EWG"]="tunemax::GroupedEightWavePipeline"
pmap["AX"]="tunemax::AsyncXorPipeline"
pmap["V3S"]="tunemax::CompV3Swizzled"
pmap["V4S"]="tunemax::CompV4Swizzled"
pmap["V3S1"]="tunemax::CompV3Swizzled128"
pmap["V4S1"]="tunemax::CompV4Swizzled128"
s = re.sub(r"(using Pipeline = )[\w:]+<Problem>", rf"\g<1>{pmap[pipe]}<Problem>", s)
s = re.sub(r"DoubleSmemBuffer = \w+;", f"DoubleSmemBuffer = {'true' if pipe in ('V4','ASYNC','PS','AX','V4S','V4S1') else 'false'};", s)
s = re.sub(r"GemmPipelineScheduler::\w+;", f"GemmPipelineScheduler::{sched};", s)
s = re.sub(r"kBlockPerCu = \d+;", f"kBlockPerCu = {bpc};", s)
s = re.sub(r"bool Persistent = \w+;", f"bool Persistent = {pers};", s)
s = re.sub(r"bool Preshuffle       = \w+;", f"bool Preshuffle       = {'true' if pipe=='PS' else 'false'};", s)
s = re.sub(r"bool CShuffleEpilogue = \w+;", f"bool CShuffleEpilogue = {'true' if epi=='cs' else 'false'};", s)
open(f,"w").write(s)
