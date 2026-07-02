# # LARGEST IN A LIST

# # a = []
# # n = int(input("How many values in list : "))

# # for i in range(n):
# #   v = int(input("Enter value : "))
# #   a.append(v)

# # a.sort()
# # print(a[-1])

# #COUNT VOWELS IN STRING

# # s = str(input("Enter String: "))
# # s = s.strip().replace(" ","").lower()

# # vowels = ['a','e','i','o','u']
# # count = 0

# # for ch in s:
# #   if(ch in vowels):
# #     count += 1

# # print(count)

# # REVERSE A STRING WO SLICE

# s = str(input("Enter String: "))

# temp = ""


import json

with open(
    "data/processed/chunks.json",
    encoding="utf-8"
) as f:
    data = json.load(f)

for p in data["parents"]:
    if p["chunk_id"] == "parent_3673":
        print(p["company"])
        print(p["text"][:300])
        break