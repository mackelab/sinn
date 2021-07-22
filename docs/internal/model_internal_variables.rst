# Internal variables used in the `Model` class

`num_tidx`:
~ Used as the *anchor* time index for computational graphs
~ Used in:
  - num_tidx:
    - returns min(tidx, unlocked_hists)
    - if there are no unlocked_hists, returns model.t0idx
    - see TODOs around num_tidx

  - convert_point_updates_to_scan:
    - line 2440:  anchor_tidx = self.get_num_tidx(histories+tuple(self.statehists))
    anchor_tidx <-> hist._num_tidx  (unlocked hists only)

  - accumulate_with_offset:
    - line 2578:  numtidx = self.num_tidx  # Raises RuntimeError if errors are not synchronized
      
  - sinnfull.AlternatedSGD:prepare_compilation
    - line 1090:  self._k_vars.k = model.time.Index(model.num_tidx)  # -> curtidx_var
      + used to construct the slices for extracting the gradient subtensor
      + => must correspond to what is substituted by curtidx_var when constructing update graphs
