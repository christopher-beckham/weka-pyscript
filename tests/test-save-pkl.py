from pyscript.pyscript import ArffToArgs

# test saving the pkl to an output file

f = ArffToArgs()
f.set_input("../datasets/iris.arff")
args = f.get_args()
f.save("iris.pkl.gz")
f.close()

# test normal

f = ArffToArgs()
f.set_input("../datasets/iris.arff")
args = f.get_args()
print f.output
f.close()