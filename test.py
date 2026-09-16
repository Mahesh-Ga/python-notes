def rain_water(heights: list) -> int:
   left =0 
   right=left+1
   sum = 0

   while right < len(heights): 
      if heights[right] > heights[left]:
        left = right
        right+=1

      sum += (heights[right]  - heights[left])
      right +=1
      left += 1
   
   return sum
heights = [0,1,0,2,1,0,1,3,2,1,2,1]
result = rain_water(heights)

print(result)