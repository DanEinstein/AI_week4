def sort_list_of_dicts(list_of_dicts, key, reverse=False):
    """
    Sorts a list of dictionaries by a specific key.

    Args:
        list_of_dicts (list): The list of dictionaries to sort.
        key (str): The dictionary key to sort by.
        reverse (bool): If True, the list is sorted in descending order.
                        Defaults to False (ascending order).

    Returns:
        list: A new list containing the sorted dictionaries.
    """
    return sorted(list_of_dicts, key=lambda item: item[key], reverse=reverse)

# Example list of dictionaries
people = [
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago'},
    {'name': 'Alice', 'age': 30, 'city': 'New York'},
    {'name': 'Bob', 'age': 25, 'city': 'Los Angeles'}
]

# 1. Sort by 'age' in ascending order (the default)
sorted_by_age = sort_list_of_dicts(people, 'age')
print("Sorted by age (ascending):")
# Expected output:
# [{'name': 'Bob', 'age': 25, 'city': 'Los Angeles'},
#  {'name': 'Alice', 'age': 30, 'city': 'New York'},
#  {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}]
for person in sorted_by_age:
    print(person)

print("\n" + "-"*20 + "\n")

# 2. Sort by 'name' in descending order
sorted_by_name_desc = sort_list_of_dicts(people, 'name', reverse=True)
print("Sorted by name (descending):")
# Expected output:
# [{'name': 'Charlie', 'age': 35, 'city': 'Chicago'},
#  {'name': 'Bob', 'age': 25, 'city': 'Los Angeles'},
#  {'name': 'Alice', 'age': 30, 'city': 'New York'}]
for person in sorted_by_name_desc:
    print(person)

# The original list remains unchanged
print("\n" + "-"*20 + "\n")
print("Original list:")
for person in people:
    print(person)
