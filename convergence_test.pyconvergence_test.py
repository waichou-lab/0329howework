import numpy as np
from solve_pde import solve_pde

def main():
    ms = [10, 20, 40, 80]
    errors = []
    
    print("網格尺寸與誤差:")
    for m in ms:
        norm2, _ = solve_pde(m)
        errors.append(norm2)
        print(f"m = {m:2d}, h = {1/m:.4f}, error = {norm2:.6e}")
    
    print("\n收斂階驗證:")
    for i in range(len(ms)-1):
        ratio = errors[i] / errors[i+1]
        print(f"h 從 {1/ms[i]:.4f} 到 {1/ms[i+1]:.4f}，誤差比 = {ratio:.2f}")
    
    # 可選：計算精確收斂階
    p = []
    for i in range(len(ms)-1):
        p_i = np.log(errors[i]/errors[i+1]) / np.log(ms[i+1]/ms[i])
        p.append(p_i)
        print(f"收斂階 p = {p_i:.3f}")

if __name__ == "__main__":
    main()