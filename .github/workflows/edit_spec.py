with open('main.spec', 'r') as f:
    content = f.read()
content = content.replace('excludes=[]', 'excludes=["torch", "tensorflow", "triton"]')
with open('main.spec', 'w') as f:
    f.write(content)
