# CC-reserved RL Case Study: S4 -> RL v2

来源图: `cc_reserved_rl_delta.png`。本页只比较 `cc-reserved` 的贪心 Pass@1，同一 `idx` 在 S4 与 RL v2 之间配对。

文件说明：

- 精选 case study: 本文档
- 完整翻转清单 final 摘要: `cc_reserved_rl_flip_cases.csv`
- 全部 549 个翻转 case 的完整两次模型回答: `cc_reserved_rl_flip_cases_full.jsonl`

## Overall Flip Accounting

| item | count |
|---|---:|
| paired cases | 4372 |
| S4 wrong -> RL right | 135 |
| S4 right -> RL wrong | 414 |
| net Pass@1 flips | -279 |

结论：RL v2 修了一部分基础数论和定积分题，但退化面更宽，尤其集中在小数算术、解析几何模板、集合逻辑、方程和组合计数。图里的整体下降不是单个 source 的偶然波动，而是 414 个 1→0 翻转压过了 135 个 0→1 翻转。

## 变对: S4 Wrong -> RL Right

| family | count |
|---|---:|
| arithmetic | 22 |
| number theory | 16 |
| quadratic | 13 |
| domain assumption | 12 |
| combinatorics | 7 |
| equation | 7 |
| word problem | 7 |
| expression rewrite | 6 |
| ratio percent | 5 |
| comparison | 4 |
| differentiation | 4 |
| matrices | 4 |

### Case 1: 变对 · idx=2485 · `integration.definite_integral_basic`

**题目**

Compute ∫_0^5 20x^4 dx.

**正确答案**

```text
12500
```

**S4 模型回答** (错误; final=`12600`)

```text
<think>
An antiderivative of 20x^4 is 4x^5 (since 20/5 = 4).
At x=5: 4×5^5 = 4×3125 = 12600.
At x=0: 4×0^5 = 0.
Subtract: 12600 - 0 = 12600.
So the definite integral is 12600.
</think>
#### \boxed{12600}
```

**RL v2 模型回答** (正确; final=`12500`)

```text
<think>
An antiderivative of 20x^4 is 4x^5 (since 20/5 = 4).
At x=5: 4×5^5 = 4×3125 = 12500.
At x=0: 4×0^5 = 0.
Subtract: 12500 - 0 = 12500.
So the definite integral is 12500.
</think>
#### \boxed{12500}
```

**观察**: RL 修复了幂函数定积分的系数/端点代入。
### Case 2: 变对 · idx=2503 · `integration.definite_integral_basic`

**题目**

Evaluate the definite integral of 20x^4 from 0 to 5.

**正确答案**

```text
12500
```

**S4 模型回答** (错误; final=`12900`)

```text
<think>
An antiderivative of 20x^4 is 4x^5 (since 20/5 = 4).
At x=5: 4×5^5 = 4×3125 = 12900.
At x=0: 4×0^5 = 0.
Subtract: 12900 - 0 = 12900.
So the definite integral is 12900.
</think>
#### \boxed{12900}
```

**RL v2 模型回答** (正确; final=`12500`)

```text
<think>
An antiderivative of 20x^4 is 4x^5 (since 20/5 = 4).
At x=5: 4×5^5 = 4×3125 = 12500.
At x=0: 4×0^5 = 0.
Subtract: 12500 - 0 = 12500.
So the definite integral is 12500.
</think>
#### \boxed{12500}
```

**观察**: RL 修复了幂函数定积分的系数/端点代入。
### Case 3: 变对 · idx=2706 · `number_theory.divisibility_rules`

**题目**

Is 371 divisible by 3?

**正确答案**

```text
No
```

**S4 模型回答** (错误; final=`Yes`)

```text
<think>
A number is divisible by 3 exactly when its digit sum is divisible by 3.
The digit sum of 371 is 12, and 12 is divisible by 3.
Therefore 371 is divisible by 3.
So the answer is Yes.
</think>
#### \boxed{Yes}
```

**RL v2 模型回答** (正确; final=`No`)

```text
<think>
A number is divisible by 3 exactly when its digit sum is divisible by 3.
The digit sum of 371 is 10, and 10 is not divisible by 3.
Therefore 371 is not divisible by 3.
So the answer is No.
</think>
#### \boxed{No}
```

**观察**: RL 的 final answer 从 Yes 校正为 No，但 reasoning 里仍可能有 digit-sum 算错或自相矛盾；这是“答案翻转正确”，不是推理完全修复。
### Case 4: 变对 · idx=2709 · `number_theory.divisibility_rules`

**题目**

Is 8742449 divisible by 3?

**正确答案**

```text
No
```

**S4 模型回答** (错误; final=`Yes`)

```text
<think>
A number is divisible by 3 exactly when its digit sum is divisible by 3.
The digit sum of 8742449 is 37, and 37 is divisible by 3.
Therefore 8742449 is divisible by 3.
So the answer is Yes.
</think>
#### \boxed{Yes}
```

**RL v2 模型回答** (正确; final=`No`)

```text
<think>
A number is divisible by 3 exactly when its digit sum is divisible by 3.
The digit sum of 8742449 is 30, and 30 is not divisible by 3.
Therefore 8742449 is not divisible by 3.
So the answer is No.
</think>
#### \boxed{No}
```

**观察**: RL 的 final answer 从 Yes 校正为 No，但 reasoning 里仍可能有 digit-sum 算错或自相矛盾；这是“答案翻转正确”，不是推理完全修复。
### Case 5: 变对 · idx=2733 · `number_theory.gcd_euclidean_algorithm`

**题目**

Find gcd(45, 120) using the Euclidean algorithm.

**正确答案**

```text
15
```

**S4 模型回答** (错误; final=`5`)

```text
<think>
Use the Euclidean algorithm on 120 and 45.
120=45×2+10.
45=10×4+5.
10=5×2+0.
The last nonzero remainder is 5, so the gcd is 5.
</think>
#### \boxed{5}
```

**RL v2 模型回答** (正确; final=`15`)

```text
<think>
Use the Euclidean algorithm on 120 and 45.
120=45×3+15.
45=15×3+0.
The last nonzero remainder is 15, so the gcd is 15.
</think>
#### \boxed{15}
```

**观察**: RL 在基本数论步骤上更稳，能按 gcd/lcm 关系给出正确值。
### Case 6: 变对 · idx=2772 · `number_theory.lcm_using_gcd`

**题目**

Find lcm(24, 29).

**正确答案**

```text
696
```

**S4 模型回答** (错误; final=`796`)

```text
<think>
First find gcd(24,29)=1.
Use lcm(a,b)=|ab|/gcd(a,b): lcm(24,29)=(24×29)/1=796.
</think>
#### \boxed{796}
```

**RL v2 模型回答** (正确; final=`696`)

```text
<think>
First find gcd(24,29)=1.
Use lcm(a,b)=|ab|/gcd(a,b): lcm(24,29)=(24×29)/1=696.
</think>
#### \boxed{696}
```

**观察**: RL 在基本数论步骤上更稳，能按 gcd/lcm 关系给出正确值。
### Case 7: 变对 · idx=2823 · `number_theory.prime_factorization`

**题目**

Find the prime factorization of 1262.

**正确答案**

```text
2 × 631
```

**S4 模型回答** (错误; final=`2 × 3 × 7 × 31`)

```text
<think>
Divide 1262 by prime numbers from small to large.
1262 is divisible by 2^1, leaving 631.
631 is divisible by 3^1, leaving 217.
217 is divisible by 7^1, leaving 31.
31 is divisible by 31^1, leaving 1.
Therefore, 1262=2 × 3 × 7 × 31.
</think>
#### \boxed{2 × 3 × 7 × 31}
```

**RL v2 模型回答** (正确; final=`2 × 631`)

```text
<think>
Divide 1262 by prime numbers from small to large.
1262 is divisible by 2^1, leaving 631.
631 is divisible by 631^1, leaving 1.
Therefore, 1262=2 × 631.
</think>
#### \boxed{2 × 631}
```

**观察**: RL 修复了质因数分解中的错误拆分。
### Case 8: 变对 · idx=2829 · `number_theory.prime_factorization`

**题目**

Find the prime factorization of 502.

**正确答案**

```text
2 × 251
```

**S4 模型回答** (错误; final=`2 × 3 × 77`)

```text
<think>
Divide 502 by prime numbers from small to large.
502 is divisible by 2^1, leaving 251.
251 is divisible by 3^1, leaving 77.
77 is divisible by 77^1, leaving 1.
Therefore, 502=2 × 3 × 77.
</think>
#### \boxed{2 × 3 × 77}
```

**RL v2 模型回答** (正确; final=`2 × 251`)

```text
<think>
Divide 502 by prime numbers from small to large.
502 is divisible by 2^1, leaving 251.
251 is divisible by 251^1, leaving 1.
Therefore, 502=2 × 251.
</think>
#### \boxed{2 × 251}
```

**观察**: RL 修复了质因数分解中的错误拆分。


## 变错: S4 Right -> RL Wrong

| family | count |
|---|---:|
| arithmetic | 82 |
| equation | 27 |
| set logic | 23 |
| expression rewrite | 20 |
| plane geometry | 20 |
| combinatorics | 19 |
| quadratic | 18 |
| comparison | 16 |
| domain assumption | 16 |
| word problem | 16 |
| analytic geometry schema | 14 |
| ratio percent | 14 |
| function property schema | 12 |
| exp log | 10 |
| sequence schema | 10 |

### Case 1: 变错 · idx=324 · `arithmetic.decimal_addition`

**题目**

Compute 55.57 + 483.1.

**正确答案**

```text
538.67
```

**S4 模型回答** (正确; final=`538.67`)

```text
<think>
Align decimal places. Use 2 decimal place(s) for both numbers.
At this scale, 55.57 becomes 5557 and 483.1 becomes 48310.
Add as integers at that scale: 5557+48310=53867.
Put the decimal point back 2 place(s) from the right, giving 538.67.
</think>
#### \boxed{538.67}
```

**RL v2 模型回答** (错误; final=`548.87`)

```text
<think>
Align decimal places. Use 2 decimal place(s) for both numbers.
At this scale, 55.57 becomes 5557 and 483.1 becomes 48310.
Add as integers at that scale: 5557+48310=54887.
Put the decimal point back 2 place(s) from the right, giving 548.87.
</think>
#### \boxed{548.87}
```

**观察**: 代表该 family 的 S4/RL 贪心首答翻转。
### Case 2: 变错 · idx=338 · `arithmetic.decimal_division_by_decimal`

**题目**

Compute 2.156 ÷ 0.28.

**正确答案**

```text
7.7
```

**S4 模型回答** (正确; final=`7.7`)

```text
<think>
Remove the decimal points and divide as whole numbers: 2.156 has 3 decimal place(s) and 0.28 has 2, so first compute the whole-number division 2156 ÷ 28.
Use long division, processing the digits of 2156 from left to right.
Bring down digit 2; the current value is 2.
Since 2<28 and no quotient digit has been written yet, wait for the next digit.
Bring down digit 1; the current value is 21.
Since 21<28 and no quotient digit has been written yet, wait for the next digit.
Bring down digit 5; the current value is 215.
215÷28: find the largest digit q with q×28≤215. 7×28=196≤215, but (7+1)×28=224>215, so q=7. Subtract: 215-196=19 (it is ≥0 and <28, so q is correct).
Bring down digit 6; the current value is 196.
196÷28: find the largest digit q with q×28≤196. 7×28=196≤196, but (7+1)×28=224>196, so q=7. Subtract: 196-196=0 (it is ≥0 and <28, so q is correct).
So 2156 ÷ 28 = 77.
The quotient has 3 - 2 = 1 decimal place(s), so place the decimal point and the result is 7.7.
</think>
#### \boxed{7.7}
```

**RL v2 模型回答** (错误; final=`0.77`)

```text
<think>
Remove the decimal points and divide as whole numbers: 2.156 has 4 decimal place(s) and 0.28 has 2, so first compute the whole-number division 2156 ÷ 28.
Use long division, processing the digits of 2156 from left to right.
Bring down digit 2; the current value is 2.
Since 2<28 and no quotient digit has been written yet, wait for the next digit.
Bring down digit 1; the current value is 21.
Since 21<28 and no quotient digit has been written yet, wait for the next digit.
Bring down digit 5; the current value is 215.
215÷28: find the largest digit q with q×28≤215. 7×28=196≤215, but (7+1)×28=224>215, so q=7. Subtract: 215-196=19 (it is ≥0 and <28, so q is correct).
Bring down digit 6; the current value is 196.
196÷28: find the largest digit q with q×28≤196. 7×28=196≤196, but (7+1)×28=224>196, so q=7. Subtract: 196-196=0 (it is ≥0 and <28, so q is correct).
So 2156 ÷ 28 = 77.
The quotient has 4 - 2 = 2 decimal place(s), so place the decimal point and the result is 0.77.
</think>
#### \boxed{0.77}
```

**观察**: 小数除法尺度退化，常差 10 倍或把除法方向/位数处理错。
### Case 3: 变错 · idx=342 · `arithmetic.decimal_division_by_decimal`

**题目**

Compute 0.072 ÷ 0.06.

**正确答案**

```text
1.2
```

**S4 模型回答** (正确; final=`1.2`)

```text
<think>
Remove the decimal points and divide as whole numbers: 0.072 has 3 decimal place(s) and 0.06 has 2, so first compute the whole-number division 72 ÷ 6.
Use long division, processing the digits of 72 from left to right.
Bring down digit 7; the current value is 7.
7÷6: find the largest digit q with q×6≤7. 1×6=6≤7, but (1+1)×6=12>7, so q=1. Subtract: 7-6=1 (it is ≥0 and <6, so q is correct).
Bring down digit 2; the current value is 12.
12÷6: find the largest digit q with q×6≤12. 2×6=12≤12, but (2+1)×6=18>12, so q=2. Subtract: 12-12=0 (it is ≥0 and <6, so q is correct).
So 72 ÷ 6 = 12.
The quotient has 3 - 2 = 1 decimal place(s), so place the decimal point and the result is 1.2.
</think>
#### \boxed{1.2}
```

**RL v2 模型回答** (错误; final=`0.12`)

```text
<think>
Remove the decimal points and divide as whole numbers: 0.072 has 4 decimal place(s) and 0.06 has 2, so first compute the whole-number division 72 ÷ 6.
Use long division, processing the digits of 72 from left to right.
Bring down digit 7; the current value is 7.
7÷6: find the largest digit q with q×6≤7. 1×6=6≤7, but (1+1)×6=12>7, so q=1. Subtract: 7-6=1 (it is ≥0 and <6, so q is correct).
Bring down digit 2; the current value is 12.
12÷6: find the largest digit q with q×6≤12. 2×6=12≤12, but (2+1)×6=18>12, so q=2. Subtract: 12-12=0 (it is ≥0 and <6, so q is correct).
So 72 ÷ 6 = 12.
The quotient has 4 - 2 = 2 decimal place(s), so place the decimal point and the result is 0.12.
</think>
#### \boxed{0.12}
```

**观察**: 小数除法尺度退化，常差 10 倍或把除法方向/位数处理错。
### Case 4: 变错 · idx=378 · `arithmetic.decimal_multiplication`

**题目**

Compute 177.4 × 96.8.

**正确答案**

```text
17172.32
```

**S4 模型回答** (正确; final=`17172.32`)

```text
<think>
Ignore the decimal points first: 177.4 becomes 1774, and 96.8 becomes 968. Multiply these as whole numbers.
Break the multiplier 968 into place-value digits, compute a partial product for each, then add them.
Partial product for the ones digit 8: compute 1774×8.
  ones: 4×8=32, write 2, carry 3
  tens: 7×8=56, plus carry 3 = 59, write 9, carry 5
  hundreds: 7×8=56, plus carry 5 = 61, write 1, carry 6
  thousands: 1×8=8, plus carry 6 = 14, write 4, carry 1
  Final carry 1 goes to the front.
Partial product for the ones digit: 14192.
Partial product for the tens digit 6: compute 1774×6.
  ones: 4×6=24, write 4, carry 2
  tens: 7×6=42, plus carry 2 = 44, write 4, carry 4
  hundreds: 7×6=42, plus carry 4 = 46, write 6, carry 4
  thousands: 1×6=6, plus carry 4 = 10, write 0, carry 1
  Final carry 1 goes to the front.
Because this digit is in the tens place, append 1 zero to 10644, giving 106440.
Partial product for the hundreds digit 9: compute 1774×9.
  ones: 4×9=36, write 6, carry 3
  tens: 7×9=63, plus carry 3 = 66, write 6, carry 6
  hundreds: 7×9=63, plus carry 6 = 69, write 9, carry 6
  thousands: 1×9=9, plus carry 6 = 15, write 5, carry 1
  Final carry 1 goes to the front.
Because this digit is in the hundreds place, append 2 zeros to 15966, giving 1596600.
Now add the 3 partial products, one at a time, column by column from the right.
Start with 14192.
Add 106440: compute 14192+106440.
  ones: 2+0=2, write 2.
  tens: 9+4=13, write 3, carry 1.
  hundreds: 1+4+1=6, write 6.
  thousands: 4+6=10, write 0, carry 1.
  ten-thousands: 1+0+1=2, write 2.
  hundred-thousands: 0+1=1, write 1.
  Running total: 120632.
Add 1596600: compute 120632+1596600.
  ones: 2+0=2, write 2.
  tens: 3+0=3, write 3.
  hundreds: 6+6=12, write 2, carry 1.
  thousands: 0+6+1=7, write 7.
  ten-thousands: 2+9=11, write 1, carry 1.
  hundred-thousands: 1+5+1=7, write 7.
  millions: 0+1=1, write 1.
  Running total: 1717232.
The factors have 1+1=2 decimal place(s) in total.
So place the decimal point 2 place(s) from the right of 1717232, giving 17172.32.
</think>
#### \boxed{17172.32}
```

**RL v2 模型回答** (错误; final=`69269`)

```text
<think>
Ignore the decimal points first: 177.4 becomes 1774, and 96.8 becomes 968. Multiply these as whole numbers.
Break the multiplier 968 into place-value digits, compute a partial product for each, then add them.
Partial product for the ones digit: 1×968=968.
Partial product for the tens digit: 7×96=692.
Partial product for the hundreds digit: 7×9=63.
Partial product for the thousands digit: 1×9=9.
Combine to get 69269.
The product is 69269.
</think>
#### \boxed{69269}
```

**观察**: 小数乘法/小数点位置退化，RL 输出常像整数拼接或少放小数点。
### Case 5: 变错 · idx=383 · `arithmetic.decimal_multiplication`

**题目**

Compute 10.1 × 19.8.

**正确答案**

```text
199.98
```

**S4 模型回答** (正确; final=`199.98`)

```text
<think>
Ignore the decimal points first: 10.1 becomes 101, and 19.8 becomes 198. Multiply these as whole numbers.
Break the multiplier 198 into place-value digits, compute a partial product for each, then add them.
Partial product for the ones digit 8: compute 101×8.
  ones: 1×8=8, write 8
  tens: 0×8=0, write 0
  hundreds: 1×8=8, write 8
Partial product for the ones digit: 808.
Partial product for the tens digit 9: compute 101×9.
  ones: 1×9=9, write 9
  tens: 0×9=0, write 0
  hundreds: 1×9=9, write 9
Because this digit is in the tens place, append 1 zero to 909, giving 9090.
Partial product for the hundreds digit 1: compute 101×1.
  ones: 1×1=1, write 1
  tens: 0×1=0, write 0
  hundreds: 1×1=1, write 1
Because this digit is in the hundreds place, append 2 zeros to 101, giving 10100.
Now add the 3 partial products, one at a time, column by column from the right.
Start with 808.
Add 9090: compute 808+9090.
  ones: 8+0=8, write 8.
  tens: 0+9=9, write 9.
  hundreds: 8+0=8, write 8.
  thousands: 0+9=9, write 9.
  Running total: 9898.
Add 10100: compute 9898+10100.
  ones: 8+0=8, write 8.
  tens: 9+0=9, write 9.
  hundreds: 8+1=9, write 9.
  thousands: 9+0=9, write 9.
  ten-thousands: 0+1=1, write 1.
  Running total: 19998.
The factors have 1+1=2 decimal place(s) in total.
So place the decimal point 2 place(s) from the right of 19998, giving 199.98.
</think>
#### \boxed{199.98}
```

**RL v2 模型回答** (错误; final=`1980`)

```text
<think>
Ignore the decimal points first: 10.1 becomes 101, and 19.8 becomes 198. Multiply these as whole numbers.
Break the multiplier 198 into place-value digits, compute a partial product for each, then add them.
Partial product for the ones digit: 1×198=198.
Partial product for the tens digit: 0×1=0.
Partial product for the hundreds digit: 1×1=1.
Add the parts: 1980.
So 10.1 × 19.8 = 1980.
</think>
#### \boxed{1980}
```

**观察**: 小数乘法/小数点位置退化，RL 输出常像整数拼接或少放小数点。
### Case 6: 变错 · idx=222 · `analytic_geometry_schema.circle_center_radius_by_completing_square`

**题目**

Find the center and radius of x^2 + y^2 - 4x - 10y  - 71 = 0.

**正确答案**

```text
center (2, 5), radius 10
```

**S4 模型回答** (正确; final=`center (2, 5), radius 10`)

```text
<think>
x^2 - 4x completes to (x - 2)^2 by adding 4.
y^2 - 10y completes to (y - 5)^2 by adding 25.
The completed form is (x - 2)^2 + (y - 5)^2 = 100.
So the circle has center (2, 5), radius 10.
</think>
#### \boxed{center (2, 5), radius 10}
```

**RL v2 模型回答** (错误; final=`center (2, 5), radius 4`)

```text
<think>
x^2 - 4x completes to (x - 2)^2 by adding 4.
y^2 - 10y completes to (y - 5)^2 by adding 25.
The completed form is (x - 2)^2 + (y - 5)^2 = 16.
So the circle has center (2, 5), radius 4.
</think>
#### \boxed{center (2, 5), radius 4}
```

**观察**: 配方框架还在，但补平方后的常数项/半径平方计算错。
### Case 7: 变错 · idx=257 · `analytic_geometry_schema.line_circle_intersection`

**题目**

Find where the line y=0 intersects the circle x^2+y^2=49.

**正确答案**

```text
(-7, 0) and (7, 0)
```

**S4 模型回答** (正确; final=`(-7, 0) and (7, 0)`)

```text
<think>
On the line y = 0, the circle x^2 + y^2 = 49 becomes x^2 = 49.
Thus x = -7 or x = 7.
So the intersection points are (-7, 0) and (7, 0).
</think>
#### \boxed{(-7, 0) and (7, 0)}
```

**RL v2 模型回答** (错误; final=`(7, 0) and (11, 0)`)

```text
<think>
On the line y = 0, the circle x^2 + y^2 = 49 becomes x^2 = 49.
The x-coordinates of the points of intersection are x = 7 and x = 11.
So the lines intersect at (7, 0) and (11, 0).
</think>
#### \boxed{(7, 0) and (11, 0)}
```

**观察**: 代入 y=0 后的根计算退化，凭空生成了不存在的交点。
### Case 8: 变错 · idx=259 · `analytic_geometry_schema.line_from_two_points`

**题目**

Find the line through (-2,-22) and (0,-10).

**正确答案**

```text
y=6x - 10
```

**S4 模型回答** (正确; final=`y=6x - 10`)

```text
<think>
The slope is (-10 - (-22))/(0 - (-2)) = 6.
Using y = mx + b with (-2,-22) gives b = -10.
So the line is y=6x - 10.
</think>
#### \boxed{y=6x - 10}
```

**RL v2 模型回答** (错误; final=`y=5x - 10`)

```text
<think>
The slope is (-10 - (-22))/(0 - (-2)) = 5.
Using y = mx + b with (-2,-22) gives b = -10.
So the line is y=5x - 10.
</think>
#### \boxed{y=5x - 10}
```

**观察**: 斜率多半能算出，截距回代错误。
### Case 9: 变错 · idx=278 · `analytic_geometry_schema.line_intersection`

**题目**

Find the intersection of y=2x - 5 and y=7x - 10.

**正确答案**

```text
(1, -3)
```

**S4 模型回答** (正确; final=`(1, -3)`)

```text
<think>
At an intersection, 2x - 5 = 7x - 10.
Solving gives x = 1.
Substitute x = 1 to get y = -3.
So the intersection is (1, -3).
</think>
#### \boxed{(1, -3)}
```

**RL v2 模型回答** (错误; final=`(0, -10)`)

```text
<think>
At an intersection, 2x - 5 = 7x - 10.
Solving gives x = 0.
Substitute x = 0 to get y = -10.
So the intersection is (0, -10).
</think>
#### \boxed{(0, -10)}
```

**观察**: 联立方程求出 x 后，y 回代错误或直接取了原式常数。
### Case 10: 变错 · idx=279 · `analytic_geometry_schema.line_intersection`

**题目**

Find the intersection of y=4x + 10 and y=3x + 9.

**正确答案**

```text
(-1, 6)
```

**S4 模型回答** (正确; final=`(-1, 6)`)

```text
<think>
At an intersection, 4x + 10 = 3x + 9.
Solving gives x = -1.
Substitute x = -1 to get y = 6.
So the intersection is (-1, 6).
</think>
#### \boxed{(-1, 6)}
```

**RL v2 模型回答** (错误; final=`(-1, 4)`)

```text
<think>
At an intersection, 4x + 10 = 3x + 9.
Solving gives x = -1.
Substitute x = -1 to get y = 4.
So the intersection is (-1, 4).
</think>
#### \boxed{(-1, 4)}
```

**观察**: 联立方程求出 x 后，y 回代错误或直接取了原式常数。
### Case 11: 变错 · idx=288 · `analytic_geometry_schema.line_intersection`

**题目**

Find the intersection of y=5x + 24 and y=6x + 27.

**正确答案**

```text
(-3, 9)
```

**S4 模型回答** (正确; final=`(-3, 9)`)

```text
<think>
At an intersection, 5x + 24 = 6x + 27.
Solving gives x = -3.
Substitute x = -3 to get y = 9.
So the intersection is (-3, 9).
</think>
#### \boxed{(-3, 9)}
```

**RL v2 模型回答** (错误; final=`(-3, 3)`)

```text
<think>
At an intersection, 5x + 24 = 6x + 27.
Solving gives x = -3.
Substitute x = -3 to get y = 3.
So the intersection is (-3, 3).
</think>
#### \boxed{(-3, 3)}
```

**观察**: 联立方程求出 x 后，y 回代错误或直接取了原式常数。


## 诊断结论

正向 case 的共同点是任务短、目标明确：整除 Yes/No、gcd/lcm、单项式定积分、质因数分解。RL 后模型更倾向执行固定规则，减少了 S4 中的离谱常数或误判。

负向 case 更成体系：`arithmetic.decimal_*` 里 S4 能稳定处理小数点，RL 后大量输出整数化、10 倍误差、或明显不符合数量级的答案；`analytic_geometry_schema.*` 里 RL 保留了解题模板，但在半径、截距、交点 y 值这类最后一步数值回代上掉分。这些退化大多不是格式问题，而是数值计算错误。
