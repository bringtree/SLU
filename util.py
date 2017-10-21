def load(src):
  temp_file = open(src, "r", encoding='utf-8')
  save_variable = temp_file.read()
  temp_file.close()
  return save_variable
