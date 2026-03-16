#!/usr/bin/env python3
"""Simulated annealing optimizer."""
import random, math, sys

def anneal(fn, x0, T=1000, cooling=0.995, min_T=0.01, step=0.5):
    x=x0[:]; best=x[:]; best_val=fn(x); current_val=best_val
    T_cur=T
    while T_cur>min_T:
        candidate=[xi+random.gauss(0,step) for xi in x]
        cand_val=fn(candidate)
        delta=cand_val-current_val
        if delta<0 or random.random()<math.exp(-delta/T_cur):
            x=candidate; current_val=cand_val
        if current_val<best_val: best=x[:]; best_val=current_val
        T_cur*=cooling
    return best, best_val

if __name__ == "__main__":
    random.seed(42)
    rosenbrock=lambda x: sum(100*(x[i+1]-x[i]**2)**2+(1-x[i])**2 for i in range(len(x)-1))
    x0=[random.uniform(-5,5) for _ in range(3)]
    best,val=anneal(rosenbrock, x0, T=5000, cooling=0.999)
    print(f"Best: {[f'{x:.4f}' for x in best]}")
    print(f"Value: {val:.6f} (optimal: 0)")
