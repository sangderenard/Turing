import pickle
x=pickle.load(open('build/full_formal_diagnostic/resolved-process-graph.pkl','rb'))
print(type(x), x)
print(getattr(x,'G',None), getattr(x,'graph',None))
print(vars(x).keys() if hasattr(x,'__dict__') else '')
