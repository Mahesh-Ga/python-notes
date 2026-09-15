1. Flatten the Array

```python
def flatten_arary(array):
    result = []
    for i in array:
        if isinstance(i,list):
            r = flatten_arary(i)
            result.extend(r)
        else:
            result.append(i) 
    return result

arr = [1, [2, [3, 4], 5], [6, 7]]
output = flatten_arary(array=arr)
print(output)
```

2. 
```
+------------------+-------------------------+------------------------+-------------------------------+
| Feature          | List                    | Tuple                  | Set                           |
+------------------+-------------------------+------------------------+-------------------------------+
| Order            | Ordered                 | Ordered                | Unordered                     | (as per insertion order)
| Mutability       | Mutable (changeable)    | Immutable (fixed)      | Mutable (can add/remove items)|
| Duplicates       | Allowed                 | Allowed                | Not allowed (unique only)     |
| Indexing/Slicing | Supported               | Supported              | Not supported                 |
| Syntax           | Square brackets `[]`    | Parentheses `()`       | Curly braces `{}`             |
+------------------+-------------------------+------------------------+-------------------------------+
```

3.  reverse a string, list , array

```python
def reverse(s):
    reverse_string = ""
    for i in range(len(s)-1, -1, -1):
        reverse_string = reverse_string + s[i]
    
    return reverse_string

text = "mahesh"
r = reverse(text)

## Slicing 
reversed_text = text[::-1]
print(reversed_text)
```

4. 2Sum 
You are given an array of integers nums and an integer `target`, return indices of the two numbers such that they add up to target.
```python

nums = [2,7,11,15]
target = 9
def solve(nums):
    for i in range(0, len(nums)):
        for j in range(i+1, len(nums)):
            if (nums[i] + nums[j]) == target: 
                return [i,j]

i = solve(nums=nums)
print(i)


## Optimized (Hash Map )
## Complement: number need to be target value 

def two_sum(nums, target):
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i
```

5. Rolling Sum: 
Input: nums = [1,2,3,4]
Output: [1,3,6,10]
```python
def runningSum(nums: list[int]) -> list[int]:
    for i in range(1, len(nums)):
        nums[i] += nums[i-1]
    return nums
```

6. get all substring 

```python
def all_substrings(s: str):
    result = []
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            result.append(s[i:j])
    return result

s = "pwwkew"
print(all_substrings(s))
```

7. return longest non repeating substring within a string

- start from left and right pointer starting from index 0 
- increment right 
- increment left when found dupplicate and increment 1 by where it is present earlier

```python
def longest_unique_substring(s: str) -> str:
    last_index = {} # char -> last seen position
    left = 0
    max_start = 0
    max_len = 0

    for right, char in enumerate(s): # (i , char)
        # if char repeated inside current window
        if char in last_index and last_index[char] >= left:
            left = last_index[char] + 1

        last_index[char] = right

        # update best
        curr_len = right - left + 1
        if curr_len > max_len:
            max_len = curr_len
            max_start = left

    return s[max_start : max_start + max_len]

```