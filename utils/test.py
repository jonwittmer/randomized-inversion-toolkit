from generate_tables import writeLatexTables

results = {'method1': {'samples': [10, 100, 1000], 'rel_error': [1, 2, 0.0347]}}
path = 'table.tex'
writeLatexTables(results, path)
