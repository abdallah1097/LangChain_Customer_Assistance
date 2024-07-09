def split_menu(filename):
  """
  Splits a menu file into multiple files based on the "$$" delimiter.

  Args:
    filename: The name of the menu file to split.
  """
  counter = 1
  current_file = None
  with open(filename, 'r') as menu_file:
    for line in menu_file:
      if line.strip() == '$$':
        # Close current file if open
        if current_file is not None:
          current_file.close()
        # Open new file
        current_file = open(f'meal_{counter}.txt', 'w')
        counter += 1
      else:
        # Write line to current file
        current_file.write(line)
  # Close the last file if open
  if current_file is not None:
    current_file.close()


# Example usage
split_menu('menu.txt')
print(f"Menu split into files: meal_[1..n].txt")