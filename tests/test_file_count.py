from src.data.validator import validate_file_count
from config import MAX_FILES


# Valid file count
result = validate_file_count(
    4,
    MAX_FILES
)

print("Valid file count:")
print(result)


# Invalid file count
result = validate_file_count(
    5,
    MAX_FILES
)

print("\nInvalid file count:")
print(result)