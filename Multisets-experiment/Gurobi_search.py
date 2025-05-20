from gurobipy import Model, GRB, quicksum


def check_all_ones(n, v_dict, m):
    mu = { x:1 for x in range(2**n) }
    # 1) total-size
    total = sum(mu.values())
    print("sum_x mu[x] =", total, "  should be", 2**m)
    # 2) per-beta checks
    for beta, v in v_dict.items():
        s = sum(
            ((-1)**((x & beta).bit_count() & 1)) * mu[x]
            for x in mu
        )
        print(f"β={beta:2d}:  LHS={s:4d}", end="")
        mod = 2**(v+1)
        if s % mod:
            print(f"  ✗ not divisible by {mod}")
        else:
            d = s // mod
            print(f"  = {mod}·{d:4d}", end="")
            print("  ✓ odd" if d%2 else "  ✗ even")

            
            
def print_constraints(model):
        # now print each constraint
    for c in model.getConstrs():
        # fetch the left‐hand side LinExpr
        lhs = model.getRow(c)
        # sense is one of '<', '>', or '='
        sense = c.Sense
        # right‐hand side
        rhs = c.RHS
        print(f"{c.ConstrName}:   {lhs} {sense} {rhs}")

    # after you’ve built the model…
    model.write("model.lp")    # human‐readable LP format
    # or
    model.write("model.mps")   # more compact MPS format


def find_multiset_gurobi(n, v_dict, m):
    """
    Solve for a multiset M in F2^n satisfying divisibility properties using Gurobi:
      - For each beta in F2^n \ {0}, sum_x (-1)^{<beta,x>} * mu[x] = 2^{v_beta+1} * odd
      - sum_x mu[x] = 2^m
    Each d_beta is enforced to be integer, then we force (2*d_beta+1) odd.

    Args:
        n (int): dimension of the F2 vector space
        v_dict (dict[int, int]): map from beta (1..2^n-1) to v_beta
        m (int): exponent for total multiset size (|M| = 2^m)

    Returns:
        (dict[int,int], dict[int,int]): solutions for mu(x) and d_beta
    """
    # Create model
    model = Model("Multiset_Divisibility_OddDBeta")
    model.Params.LogToConsole = 1  # set to 0 to suppress solve output

    # mu[x] >= 0 integer
    mu = model.addVars(range(2**n), lb=0, vtype=GRB.INTEGER, name="mu")
    # d_beta integer
    d   = model.addVars(range(1,2**n), vtype=GRB.INTEGER, name="d")
    # z_beta binary switch
    z   = model.addVars(range(1,2**n), vtype=GRB.BINARY,  name="z")

    # total-size
    model.addConstr(quicksum(mu[x] for x in mu) == 2**m, "TotalSize")

    for beta in range(1, 2**n):
        v = v_dict.get(beta, 0)

        # build S_beta
        S = quicksum(
            ((-1)**((x & beta).bit_count() & 1)) * mu[x]
            for x in mu
        )

        # S_beta == 2^(v+1)*(2*d_beta+1)*z_beta
        if v==0:
            model.addConstr(
            S == 2**(v+1) * (2*d[beta] + 1),
            name=f"Walsh_beta_{beta}"
            )
        elif v==m-1:
            model.addConstr(
            S == 0,
            name=f"Walsh_beta_{beta}"
            )
        else: 
            model.addConstr(
            S == 2**(v+1) * (2*d[beta] + 1),
            name=f"Walsh_beta_{beta}"
            )
        # Optionally, you could also force that if z=1 then
        # (2*d+1) is odd, but 2*d+1 is always odd for integer d,
        # so no extra constraint is needed there.

    # minimize the maximum mu[x] as before
    T = model.addVar(lb=0, vtype=GRB.INTEGER, name="T")
    for x in mu:
        model.addConstr(mu[x] <= T, name=f"Upper_mu_{x}")
    model.setObjective(T, GRB.MINIMIZE)
    
    #     # We just want feasibility, so give it a zero objective
#     model.setObjective(0, GRB.MINIMIZE)

    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ValueError("No optimal solution")

    mu_sol = {x: int(mu[x].X) for x in mu}
    d_sol  = {b: int(d[b].X)   for b in d}
    zeros  = [b for b in z if z[b].X < 0.5]

    return mu_sol, d_sol, zeros

if __name__ == "__main__":
    # Example usage for n=4
    n = 4
    v_dict = {}
    m = n 
    for i in range(1,16):
        if i == 8:
            v_dict[i]=m-2
            continue
        v_dict[i]= m-1
        
    
   
    check_all_ones(n, v_dict, m)
    mu_solution, d_solution, z_solution = find_multiset_gurobi(n, v_dict, m)
    print("Mu(x):", mu_solution)
    print("Odd d_beta:", d_solution)
