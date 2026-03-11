// Copyright 2026 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

package tl_test_pkg;

  // SECDED Functions for Integrity
  function automatic logic [6:0] secded_inv_39_32_enc(logic [31:0] data);
    logic [6:0] ecc;
    ecc[0] = ^(data & 32'h2606BD25);
    ecc[1] = ^(data & 32'hDEBA8050);
    ecc[2] = ^(data & 32'h413D89AA);
    ecc[3] = ^(data & 32'h31234ED1);
    ecc[4] = ^(data & 32'hC2C1323B);
    ecc[5] = ^(data & 32'h2DCC624C);
    ecc[6] = ^(data & 32'h98505586);
    return ecc ^ 7'h2A;
  endfunction

  function automatic logic [6:0] secded_inv_64_57_enc(logic [56:0] data);
    logic [6:0] ecc;
    ecc[0] = ^(data & 57'h0103FFF800007FFF);
    ecc[1] = ^(data & 57'h017C1FF801FF801F);
    ecc[2] = ^(data & 57'h01BDE1F87E0781E1);
    ecc[3] = ^(data & 57'h01DEEE3B8E388E22);
    ecc[4] = ^(data & 57'h01EF76CDB2C93244);
    ecc[5] = ^(data & 57'h01F7BB56D5525488);
    ecc[6] = ^(data & 57'h01FBDDA769A46910);
    return ecc ^ 7'h2a;
  endfunction

endpackage
