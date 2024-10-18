import os

figoutpath_base = './'
# figoutpath_base = '/Users/hansem/Dropbox (MIT)/MPong/figs/mpong_phys/redo_paper_scratch_202202' # '/om/user/rishir/figs/mpong_phys/paper_scratch_202202/'

if os.path.isdir(figoutpath_base) is False:
    os.makedirs(figoutpath_base)
