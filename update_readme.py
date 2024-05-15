import pandas as pd

df = pd.read_csv('results/results.csv')

table_md = df.to_markdown(index=False)

with open('README.md', 'r') as file:
    readme_content = file.read()

start_marker = '<!-- results table start -->'
end_marker = '<!-- results table end -->'

start_index = readme_content.find(start_marker) + len(start_marker)
end_index = readme_content.find(end_marker)

updated_readme = readme_content[:start_index] + '\n' + table_md + '\n' + readme_content[end_index:]

with open('README.md', 'w') as file:
    file.write(updated_readme)