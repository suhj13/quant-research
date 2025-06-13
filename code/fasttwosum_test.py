import numpy as np
import pandas as pd

# Utillize float16 as inputs
def fl(x):
    y = np.float16(x)
    return y

# Fasttwosum algorithm given |a| >= |b|
def fasttwosum(a, b):
    a = fl(a)
    b = fl(b)
    s = a + b   # sum in float16
    t = s - a   # lost portion of b due to rounding 
    error = b - t   # error term
    return s, error

def test(num_tests = 10000, tolerance = 1e-3):
    failures = []

    a_vals = np.random.uniform(-1, 1, num_tests).astype(np.float16)
    b_vals = np.random.uniform(-1, 1, num_tests).astype(np.float16)

    for i in range(num_tests):
        a = a_vals[i]
        b = b_vals[i]

        if abs(a) < abs(b):
            a, b = b, a
        
        s16, error16 = fasttwosum(a, b)

        a32 = np.float32(a)
        b32 = np.float32(b)
        accurate_sum = a32 + b32

        calc_sum = np.float32(s16) + np.float32(error16)

        if not np.isclose(calc_sum, accurate_sum, atol = tolerance, rtol = 0):
            failures.append({
                'index': i,
                'a': fl(a),
                'b': fl(b),
                's16': float(s16),
                'err16': float(error16),
                'accurate sum': float(accurate_sum),
                'calculated sum': float(calc_sum),
                'difference': float(calc_sum - accurate_sum)
            })

    return {
        'total_tests': num_tests,
        'num_failures': len(failures),
        'failures': failures[:5]
    }

edge_cases = [
    (1.0,    1e-4),
    (1.0,    1e-2),
    (0.125,  0.125),
    (0.1,   -0.1),
    (32700.0,1.0), 
    (1e-4,   1e-4)
]

edge_results = []
for a, b in edge_cases:
    # Enforce |a| >= |b|
    if abs(b) > abs(a):
        a, b = b, a
    s16, error16 = fasttwosum(a, b)
    a32, b32 = np.float32(a), np.float32(b)
    accurate_sum = a32 + b32
    calc_sum = np.float32(s16) + np.float32(error16)
    edge_results.append({
        'a': fl(a),
        'b': fl(b),
        's16': float(s16),
        'err16': float(error16),
        'accurate sum': float(accurate_sum),
        'calculated sum': float(calc_sum),
        'difference': float(calc_sum - accurate_sum)
    })

edge_df = pd.DataFrame(edge_results)

random_test_summary = test(10000)

print("=== Edge Case Results ===")
print(edge_df.to_string(index=False))

print("\n=== Random Test Summary ===")
print(f"Total tests run: {random_test_summary['total_tests']}")
print(f"Number of failures : {random_test_summary['num_failures']}")
if random_test_summary['num_failures'] > 0:
    rand_df = pd.DataFrame(random_test_summary['failures'])
    print("\nFirst few failures:")
    print(rand_df.to_string(index=False))