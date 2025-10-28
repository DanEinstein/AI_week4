#For this program i am writting the code to sort a list of dictionaries by age and name.
users = [
    {'name': 'Alice', 'age': 30, 'city': 'Nairobi'},
    {'name': 'Bob', 'age': 22, 'city': 'Mombasa'},
    {'name': 'Charlie', 'age': 45, 'city': 'Kisumu'}
]
sorted_users = sorted(users, key=lambda item: item['age'])

sorted_by_name = sorted(users, key=lambda item: item['name'])
print(sorted_users)
print(sorted_by_name)
