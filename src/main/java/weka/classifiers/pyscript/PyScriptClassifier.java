package weka.classifiers.pyscript;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.BatchPredictor;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.WekaException;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.python.PythonSession;

/**
 * WEKA class that calls an arbitrary Python script that can take
 * training and testing instances (in the form of Numpy arrays)
 * and return predictions.
 * @author cjb60
 *
 */
public class PyScriptClassifier extends AbstractClassifier implements
	  BatchPredictor, CapabilitiesHandler {
	
	private static final long serialVersionUID = 2846535265976949760L;
	
	/**
	 * Default values for the parameters.
	 */
	private final String DEFAULT_PYFILE = "/Users/cjb60/github/ordinal-nn/train-nn.py";
	private final String DEFAULT_TRAIN_PYFILE_PARAMS = "['train', None, 'default']";
	private final String DEFAULT_TEST_PYFILE_PARAMS = "['test', None, None, 'default']";
	
	private Instances m_trainingData = null;
	private Instances m_validData = null;
	
	private boolean m_debug = false;
	
	private Filter m_nominalToBinary = null;
	private Filter m_replaceMissing = null;
	
	private String m_pickledModel = null;
	
	/** The default Python script to execute */
	private String m_pyTrainFile = DEFAULT_PYFILE;
	
	/** If there are any parameters to pass to the training script */
	private String m_pyTrainFileParams = DEFAULT_TRAIN_PYFILE_PARAMS;
	
	/** If there are any parameters to pass to the testing script */
	private String m_pyTestFileParams = DEFAULT_TEST_PYFILE_PARAMS;
	
	public String getPythonFile() {
		return m_pyTrainFile;
	}
	
	public void setPythonFile(String pyFile) {
		m_pyTrainFile = pyFile;
	}
	
	public String getTrainPythonFileParams() {
		return m_pyTrainFileParams;
	}
	
	public void setTrainPythonFileParams(String pyTrainFileParams) {
		m_pyTrainFileParams = pyTrainFileParams;
	}
	
	public String getTestPythonFileParams() {
		return m_pyTestFileParams;
	}
	
	public void setTestPythonFileParams(String pyTestFileParams) {
		m_pyTestFileParams = pyTestFileParams;
	}
	
	private void pushArgs(PythonSession session) throws Exception {
	    session.executeScript("args['num_classes'] = " + m_trainingData.numClasses(), m_debug);
	    session.executeScript("args['num_attributes'] = " + m_trainingData.numAttributes(), m_debug);
	    session.executeScript("args['num_instances'] = " + m_trainingData.numInstances(), m_debug);

	    String[] extraParams = getTrainPythonFileParams().split(",");
	    for(String param : extraParams) {
	    	String[] paramSplit = param.split("=");
	    	session.executeScript("args[" + paramSplit[0] + "] = " + paramSplit[1], m_debug);
	    }
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
	    if (!PythonSession.pythonAvailable()) {
	        // try initializing
	        if (!PythonSession.initSession("python", m_debug)) {
	          String envEvalResults = PythonSession.getPythonEnvCheckResults();
	          throw new Exception("Was unable to start python environment: "
	            + envEvalResults);
	        }
	    }
	    
	    PythonSession session = PythonSession.acquireSession(this);
		
		/*
		 * Prepare training data for script
		 */
		
		m_replaceMissing = new ReplaceMissingValues();
		m_replaceMissing.setInputFormat(data);
		data = Filter.useFilter(data, m_replaceMissing);
		
	    m_nominalToBinary = new NominalToBinary();
	    m_nominalToBinary.setInputFormat(data);
	    data = Filter.useFilter(data, m_nominalToBinary);
	    
	    /*
	     * Ok, split the data up into a training set and
	     * validation set. Whatever the training set is now,
	     * 75% of it will be training and 25% will be valid.
	     */
	    
	    Filter stratifiedFolds = new StratifiedRemoveFolds();
	    stratifiedFolds.setInputFormat(data);
	    // seed = 0, invert = true, num folds = 4, fold number = 1
	    stratifiedFolds.setOptions(new String[] {"-S","0",  "-V",  "-N","4",  "-F","1" } );
	    m_trainingData = Filter.useFilter(data, stratifiedFolds);
	    // this time, don't invert the selection
	    stratifiedFolds.setOptions( new String[] {"-S","0",  "-N","4",  "-F","1" } );
	    m_validData = Filter.useFilter(data, stratifiedFolds);    
	    
	    session.executeScript("args = dict()", m_debug);
	    
	    /*
	     * Ok, push the training data to Python. The variables will be called
	     * X and Y, so let's execute to script to rename these.
	     */
	    session.instancesToPythonAsScikietLearn(m_trainingData, "train", m_debug);
	    session.executeScript("args['X_train'] = X\nargs['y_train']=Y\n", m_debug);
	    
	    /*
	     * Push the validation data.
	     */
	    session.instancesToPythonAsScikietLearn(m_validData, "valid", m_debug);
	    session.executeScript("args['X_valid'] = X\nargs['y_valid']=Y\n", m_debug);
	    
	    System.out.format("train, valid = %s, %s\n",
	    		m_trainingData.numInstances(), m_validData.numInstances());
	    

	    pushArgs(session);
	    
	    System.out.println("Number of classes: " + m_trainingData.numClasses());
	    System.out.println("Number of attributes: " + m_trainingData.numAttributes());
	    System.out.println("Number of instances: " + m_trainingData.numInstances());
	   
	    
	    /*
	     * Build the classifier.
	     */
	    
	    String driver = "import imp\n"
	    		+ "cls = imp.load_source('train','" + getPythonFile() + "')\n" 
	    		+ "best_weights = cls.train(args)";
	    
	    //String pyFile = loadFile( getPythonFile() );
	    List<String> outAndErr = session.executeScript(driver, true);
	    
	    System.out.println(outAndErr.get(0));
	    
	    /*
	     * Now save the model parameters.
	     */
	    m_pickledModel = session.getVariableValueFromPythonAsPickledObject("best_weights", true);
	    
	    PythonSession.releaseSession(this);

	}

	@Override
	public void setBatchSize(String size) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public String getBatchSize() {
		// TODO Auto-generated method stub
		return null;
	}
	
	private String loadFile(String filename) throws IOException {
		List<String> contents = Files.readAllLines(Paths.get(filename));
		StringBuffer sb = new StringBuffer();
		for(String line : contents) {
			sb.append(line + "\n");
		}
		return sb.toString();
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
	
		Instances temp = new Instances(instance.dataset(), 0);
		temp.add(instance);
		
		return distributionsForInstances(temp)[0];
	}

	@Override
	@SuppressWarnings("unchecked")
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {
		
	    if (!PythonSession.pythonAvailable()) {
	        // try initializing
	        if (!PythonSession.initSession("python", m_debug)) {
	          String envEvalResults = PythonSession.getPythonEnvCheckResults();
	          throw new Exception("Was unable to start python environment: "
	            + envEvalResults);
	        }
	    }
	    
	    try {  
	    	
	    	insts = Filter.useFilter(insts, m_replaceMissing);
	        insts = Filter.useFilter(insts, m_nominalToBinary);
	        
	        int numClasses = insts.numClasses();
	        
	        // remove the class attribute
	        Remove r = new Remove();
	        r.setAttributeIndices("" + (insts.classIndex() + 1));
	        r.setInputFormat(insts);
	        insts = Filter.useFilter(insts, r);
	        insts.setClassIndex(-1);
	    	
		    PythonSession session = PythonSession.acquireSession(this);
		    
		    session.executeScript("args = dict()", m_debug);
		    pushArgs(session);
		   
		    
		    /*
		     * Push the test data. These will also be X and Y, so have a
		     * script that renames these to X_test and y_test.
		     */
		    session.instancesToPythonAsScikietLearn(insts, "test", m_debug);
		    session.executeScript("args['X_test'] = X\n", m_debug);
		    
		    /*
		     * Push the weights of the saved model over.
		     */
		    session.setPythonPickledVariableValue("best_weights", m_pickledModel, true);
		    
		    System.out.format("test = %s\n", insts.numInstances());
		    
		    
		    String driver = "import imp\n"
		    		+ "cls = imp.load_source('test','" + getPythonFile() + "')\n" 
		    		+ "preds = cls.test(args, best_weights)";
		    
		    List<String> outAndErr = session.executeScript(driver, true);
		    System.out.println(outAndErr.get(0));
		    
		    
		    double[][] distributions = new double[insts.numInstances()][numClasses];
		    
			List<Object> preds = 
		    	(List<Object>) session.getVariableValueFromPythonAsJson("preds", m_debug);
		    for(int y = 0; y < insts.numInstances(); y++) {
		    	Object vector = preds.get(y);
		    	double[] probs = new double[numClasses];
				List<Double> probsForThisInstance = (List<Double>) vector;
		    	for(int x = 0; x < probs.length; x++) {
		    		probs[x] = probsForThisInstance.get(x);
		    	}
		    	distributions[y] = probs;
		    }
			
			return distributions;
			
	    } catch(Exception ex) {
			ex.printStackTrace();
		} finally {
			PythonSession.releaseSession(this);
		}
	    
	    return null;
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		String tmp = Utils.getOption("fn", options);
		if(tmp.length() != 0) { 
			setPythonFile(tmp);
		}
		tmp = Utils.getOption("xp", options);
		if(tmp.length() != 0) {
			setTrainPythonFileParams(tmp);
		}
		tmp = Utils.getOption("yp", options);
		if(tmp.length() != 0) {
			setTestPythonFileParams(tmp);
		}
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		result.add("-fn");
		result.add( "" + getPythonFile() );
		result.add("-xp");
		result.add( "" + getTrainPythonFileParams() );
		result.add("-yp");
		result.add( "" + getTestPythonFileParams() );
		
		Collections.addAll(result, super.getOptions());
	    return result.toArray(new String[result.size()]);
	}
	
	public static void main(String[] argv) {
		runClassifier(new PyScriptClassifier(), argv);
	}

}
