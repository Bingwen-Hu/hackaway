"""
More formally, a monoid is an ordered pair (S, ⊗) such that S is a set and ⊗
is a binary operator, satisfying these conditions:
1. For all a and b in S, a ⊗ b is defined and is also in S.
2. For all a, b and c in S, (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c).
3. There is an element e in S such that, for all a in S, e ⊗ a = a ⊗ e = a.
Then we also say that S is a monoid under ⊗, with identity e

For example: 
• The mathematical integers are a monoid under addition, with identity 0.
They are also a monoid under multiplication, with identity 1. Both operators
are also commutative. Integers in Python have these same properties.
• The mathematical real numbers are a monoid under addition, with identity
0. They are also a monoid under multiplication, with identity 1. Both
operators are commutative.
The Python floating-point numbers are not quite a monoid under addition:
for floating-point operands, (a + b) + c is often not exactly equal to
a + (b + c) because of roundoff error. The same is true of multiplication.
But in many applications the equalities are close enough for computational
purposes.
• Python Booleans are a monoid under and, with identity True. They are also
a monoid under or, with identity False.
• Suppose that max is the maximum-function, so that max(x,y) is defined to
be x if x ≥ y and y otherwise. Recall that we can speak of max as if it were
an operator and write x max y instead of max(x,y). Then max is both
associative and commutative, and the non-negative integers are a monoid
under max, with identity 0.
"""
