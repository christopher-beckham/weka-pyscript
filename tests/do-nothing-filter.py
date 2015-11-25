from pyscript.pyscript import instance_to_string, get_header

def train(args):
    return ""

def process(args, model):
    buf = [ get_header(args) ]
    X = args["X"]
    y = args["y"] if "y" in args else None
    for i in range(0, X.shape[0]):
        if y != None:
            buf.append( instance_to_string(X[i], y[i], args) )
        else:
            buf.append( instance_to_string(X[i], None, args) )
    return "\n".join(buf)
