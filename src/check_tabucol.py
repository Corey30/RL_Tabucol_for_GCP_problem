import tabucol

print('Tabucol module attributes:', [attr for attr in dir(tabucol) if not attr.startswith('_')])
print('\nModule docstring:', tabucol.__doc__)
