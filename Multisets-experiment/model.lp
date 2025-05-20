\ Model Multiset_Divisibility_OddDBeta
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x4db53b27faa6bf98
Minimize
  0 T
Subject To
 TotalSize: mu[0] + mu[1] + mu[2] + mu[3] + mu[4] + mu[5] + mu[6] + mu[7]
   + mu[8] + mu[9] + mu[10] + mu[11] + mu[12] + mu[13] + mu[14] + mu[15]
   = 32
 Divisible_beta_1: mu[0] - mu[1] + mu[2] - mu[3] + mu[4] - mu[5] + mu[6]
   - mu[7] + mu[8] - mu[9] + mu[10] - mu[11] + mu[12] - mu[13] + mu[14]
   - mu[15] - 32 d[1] = 16
 Divisible_beta_2: mu[0] + mu[1] - mu[2] - mu[3] + mu[4] + mu[5] - mu[6]
   - mu[7] + mu[8] + mu[9] - mu[10] - mu[11] + mu[12] + mu[13] - mu[14]
   - mu[15] - 32 d[2] = 16
 Divisible_beta_3: mu[0] - mu[1] - mu[2] + mu[3] + mu[4] - mu[5] - mu[6]
   + mu[7] + mu[8] - mu[9] - mu[10] + mu[11] + mu[12] - mu[13] - mu[14]
   + mu[15] - 32 d[3] = 16
 Divisible_beta_4: mu[0] + mu[1] + mu[2] + mu[3] - mu[4] - mu[5] - mu[6]
   - mu[7] + mu[8] + mu[9] + mu[10] + mu[11] - mu[12] - mu[13] - mu[14]
   - mu[15] - 32 d[4] = 16
 Divisible_beta_5: mu[0] - mu[1] + mu[2] - mu[3] - mu[4] + mu[5] - mu[6]
   + mu[7] + mu[8] - mu[9] + mu[10] - mu[11] - mu[12] + mu[13] - mu[14]
   + mu[15] - 32 d[5] = 16
 Divisible_beta_6: mu[0] + mu[1] - mu[2] - mu[3] - mu[4] - mu[5] + mu[6]
   + mu[7] + mu[8] + mu[9] - mu[10] - mu[11] - mu[12] - mu[13] + mu[14]
   + mu[15] - 32 d[6] = 16
 Divisible_beta_7: mu[0] - mu[1] - mu[2] + mu[3] - mu[4] + mu[5] + mu[6]
   - mu[7] + mu[8] - mu[9] - mu[10] + mu[11] - mu[12] + mu[13] + mu[14]
   - mu[15] - 32 d[7] = 16
 Divisible_beta_8: mu[0] + mu[1] + mu[2] + mu[3] + mu[4] + mu[5] + mu[6]
   + mu[7] - mu[8] - mu[9] - mu[10] - mu[11] - mu[12] - mu[13] - mu[14]
   - mu[15] - 32 d[8] = 16
 Divisible_beta_9: mu[0] - mu[1] + mu[2] - mu[3] + mu[4] - mu[5] + mu[6]
   - mu[7] - mu[8] + mu[9] - mu[10] + mu[11] - mu[12] + mu[13] - mu[14]
   + mu[15] - 32 d[9] = 16
 Divisible_beta_10: mu[0] + mu[1] - mu[2] - mu[3] + mu[4] + mu[5] - mu[6]
   - mu[7] - mu[8] - mu[9] + mu[10] + mu[11] - mu[12] - mu[13] + mu[14]
   + mu[15] - 32 d[10] = 16
 Divisible_beta_11: mu[0] - mu[1] - mu[2] + mu[3] + mu[4] - mu[5] - mu[6]
   + mu[7] - mu[8] + mu[9] + mu[10] - mu[11] - mu[12] + mu[13] + mu[14]
   - mu[15] - 32 d[11] = 16
 Divisible_beta_12: mu[0] + mu[1] + mu[2] + mu[3] - mu[4] - mu[5] - mu[6]
   - mu[7] - mu[8] - mu[9] - mu[10] - mu[11] + mu[12] + mu[13] + mu[14]
   + mu[15] - 32 d[12] = 16
 Divisible_beta_13: mu[0] - mu[1] + mu[2] - mu[3] - mu[4] + mu[5] - mu[6]
   + mu[7] - mu[8] + mu[9] - mu[10] + mu[11] + mu[12] - mu[13] + mu[14]
   - mu[15] - 32 d[13] = 16
 Divisible_beta_14: mu[0] + mu[1] - mu[2] - mu[3] - mu[4] - mu[5] + mu[6]
   + mu[7] - mu[8] - mu[9] + mu[10] + mu[11] + mu[12] + mu[13] - mu[14]
   - mu[15] - 32 d[14] = 16
 Divisible_beta_15: mu[0] - mu[1] - mu[2] + mu[3] - mu[4] + mu[5] + mu[6]
   - mu[7] - mu[8] + mu[9] + mu[10] - mu[11] + mu[12] - mu[13] - mu[14]
   + mu[15] - 32 d[15] = 16
 Upper_mu_0: mu[0] <= 4
 Upper_mu_1: mu[1] <= 4
 Upper_mu_2: mu[2] <= 4
 Upper_mu_3: mu[3] <= 4
 Upper_mu_4: mu[4] <= 4
 Upper_mu_5: mu[5] <= 4
 Upper_mu_6: mu[6] <= 4
 Upper_mu_7: mu[7] <= 4
 Upper_mu_8: mu[8] <= 4
 Upper_mu_9: mu[9] <= 4
 Upper_mu_10: mu[10] <= 4
 Upper_mu_11: mu[11] <= 4
 Upper_mu_12: mu[12] <= 4
 Upper_mu_13: mu[13] <= 4
 Upper_mu_14: mu[14] <= 4
 Upper_mu_15: mu[15] <= 4
Bounds
Generals
 mu[0] mu[1] mu[2] mu[3] mu[4] mu[5] mu[6] mu[7] mu[8] mu[9] mu[10] mu[11]
 mu[12] mu[13] mu[14] mu[15] d[1] d[2] d[3] d[4] d[5] d[6] d[7] d[8] d[9]
 d[10] d[11] d[12] d[13] d[14] d[15] T
End
