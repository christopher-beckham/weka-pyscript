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
import weka.filters.unsupervised.attribute.Standardize;
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
	private final String DEFAULT_PYFILE = "/Users/cjb60/github/ordinal-nn/train.py";
	private final String DEFAULT_TRAIN_PYFILE_PARAMS = 
			"'num_hidden_units'=10,'epochs'=100,'batch_size'=128,'lambda'=0.001,'alpha'=0.01,'kappa'=True";
	private final String DEFAULT_TEST_PYFILE_PARAMS = DEFAULT_TRAIN_PYFILE_PARAMS;
	
	private final boolean DEFAULT_SHOULD_STANDARDIZE = false;
	private final boolean DEFAULT_SHOULD_BINARIZE = false;
	private final boolean DEFAULT_SHOULD_IMPUTE = false;
	private final boolean DEFAULT_USE_VALIDATION_SET = false;
	
	private boolean m_shouldStandardize = DEFAULT_SHOULD_STANDARDIZE;
	private boolean m_shouldBinarize = DEFAULT_SHOULD_BINARIZE;
	private boolean m_shouldImpute = DEFAULT_SHOULD_IMPUTE;
	private boolean m_useValidationSet = DEFAULT_USE_VALIDATION_SET;
	
	private Instances m_trainingData = null;
	private Instances m_validData = null;
	
	private boolean m_debug = false;
	
	private Filter m_nominalToBinary = null;
	private Filter m_standardize = null;
	private Filter m_replaceMissing = null;
	
	private String m_pickledModel = null;
	
	private PythonSession m_session = null;
	
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
	
	public boolean getShouldStandardize() {
		return m_shouldStandardize;
	}
	
	public void setShouldStandardize(boolean b) {
		m_shouldStandardize = b;
	}
	
	public boolean getShouldBinarize() {
		return m_shouldBinarize;
	}
	
	public void setShouldBinarize(boolean b) {
		m_shouldBinarize = b;
	}
	
	public boolean getShouldImpute() {
		return m_shouldImpute;
	}
	
	public void setShouldImpute(boolean b) {
		m_shouldImpute = b;
	}
	
	public boolean getUseValidationSet() {
		return m_useValidationSet;
	}
	
	public void setUseValidationSet(boolean b) {
		m_useValidationSet = b;
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
		
		initPythonSession();
	    
		
		/*
		 * Prepare training data for script
		 */
	    if( getShouldImpute() ) {
	    	m_replaceMissing = new ReplaceMissingValues();
			m_replaceMissing.setInputFormat(data);
			data = Filter.useFilter(data, m_replaceMissing);
	    }
		if( getShouldBinarize() ) {
			m_nominalToBinary = new NominalToBinary();
	    	m_nominalToBinary.setInputFormat(data);
	    	data = Filter.useFilter(data, m_nominalToBinary);
		}
		if( getShouldStandardize() ) {
			m_standardize = new Standardize();
			m_standardize.setInputFormat(data);
			data = Filter.useFilter(data, m_standardize);
		}
	    
	    /*
	     * Ok, split the data up into a training set and
	     * validation set. Whatever the training set is now,
	     * 75% of it will be training and 25% will be valid.
	     */	    
		if( getUseValidationSet() ) {
			System.err.println("Use validation set");
		    Filter stratifiedFolds = new StratifiedRemoveFolds();
		    stratifiedFolds.setInputFormat(data);
		    // seed = 0, invert = true, num folds = 4, fold number = 1
		    stratifiedFolds.setOptions(new String[] {"-S","0",  "-V",  "-N","4",  "-F","1" } );
		    m_trainingData = Filter.useFilter(data, stratifiedFolds);
		    // this time, don't invert the selection
		    stratifiedFolds.setOptions( new String[] {"-S","0",  "-N","4",  "-F","1" } );
		    m_validData = Filter.useFilter(data, stratifiedFolds);
		} else {
			m_trainingData = data;
		}
	    
	    m_session.executeScript("args = dict()", m_debug);
	    
	    /*
	     * Ok, push the training data to Python. The variables will be called
	     * X and Y, so let's execute to script to rename these.
	     */
	    m_session.instancesToPythonAsScikietLearn(m_trainingData, "train", m_debug);
	    m_session.executeScript("args['X_train'] = X\nargs['y_train']=Y\n", m_debug);
	    
	    /*
	     * Push the validation data.
	     */
	    if( getUseValidationSet() ) {
	    	m_session.instancesToPythonAsScikietLearn(m_validData, "valid", m_debug);
	    	m_session.executeScript("args['X_valid'] = X\nargs['y_valid']=Y\n", m_debug);
	    }
	    
	    //System.out.format("train, valid = %s, %s\n",
	    //		m_trainingData.numInstances(), m_validData.numInstances());
	    

	    pushArgs(m_session);
	    
	    System.out.println("Number of classes: " + m_trainingData.numClasses());
	    System.out.println("Number of attributes: " + m_trainingData.numAttributes());
	    System.out.println("Number of instances: " + m_trainingData.numInstances());
	   
	    
	    /*
	     * Build the classifier.
	     */
	    
	    String driver = "best_weights = cls.train(args)";
	    
	    //String pyFile = loadFile( getPythonFile() );
	    List<String> outAndErr = m_session.executeScript(driver, true);
	    
	    System.out.println(outAndErr.get(0));
	    
	    /*
	     * Now save the model parameters.
	     */
	    m_pickledModel = m_session.getVariableValueFromPythonAsPickledObject("best_weights", true);
	    
	    //PythonSession.releaseSession(this);

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
	
	private void initPythonSession() throws Exception {
	    if (!PythonSession.pythonAvailable()) {
	        // try initializing
	        if (!PythonSession.initSession("python", m_debug)) {
	          String envEvalResults = PythonSession.getPythonEnvCheckResults();
	          throw new Exception("Was unable to start python environment: "
	            + envEvalResults);
	        } else {
	        	// success
	        }
	    }
    	
    	// success!
    	if(m_session == null) {
    		m_session = PythonSession.acquireSession(this);
    		
    		System.err.println("This should only run once per x-val");
    	
	    	// now load training and testing class
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + getPythonFile() + "')\n";
	    	m_session.executeScript(driver, m_debug);
    	}
	    
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public double[] distributionForInstance(Instance inst)
			throws Exception {
		
		Instances insts = new Instances(inst.dataset(), 0);
		insts.add(inst);
		
	    initPythonSession();
	    
	    try {
	    	
		    if( getShouldImpute() ) {
		    	insts = Filter.useFilter(insts, m_replaceMissing);
		    }
			if( getShouldBinarize() ) {
				insts = Filter.useFilter(insts, m_nominalToBinary);
			}
			if( getShouldStandardize() ) {
				insts = Filter.useFilter(insts, m_standardize);
			}
	        
	        int numClasses = insts.numClasses();
	        
	        // remove the class attribute
	        Remove r = new Remove();
	        r.setAttributeIndices("" + (insts.classIndex() + 1));
	        r.setInputFormat(insts);
	        insts = Filter.useFilter(insts, r);
	        insts.setClassIndex(-1);
		    
		    m_session.executeScript("args = dict()", m_debug);
		    pushArgs(m_session);
		   
		    
		    /*
		     * Push the test data. These will also be X and Y, so have a
		     * script that renames these to X_test and y_test.
		     */
		    m_session.instancesToPythonAsScikietLearn(insts, "test", m_debug);
		    m_session.executeScript("args['X_test'] = X\n", m_debug);
		    
		    /*
		     * Push the weights of the saved model over.
		     */
		    m_session.setPythonPickledVariableValue("best_weights", m_pickledModel, true);
		    
		    System.out.format("test = %s\n", insts.numInstances());
		    
		    
		    String driver = "preds = cls.test(args, best_weights)";
		    
		    List<String> outAndErr = m_session.executeScript(driver, true);
		    System.out.println(outAndErr.get(0));
		    
		    
		    //double[] distribution = new double[numClasses];
		    
			List<Object> preds = 
		    	(List<Object>) m_session.getVariableValueFromPythonAsJson("preds", m_debug);

	    	Object vector = preds.get(0);
	    	double[] probs = new double[numClasses];
			List<Double> probsForThisInstance = (List<Double>) vector;
	    	for(int x = 0; x < probs.length; x++) {
	    		probs[x] = probsForThisInstance.get(x);
	    	}
	    	
	    	return probs;
			
	    } catch(Exception ex) {
			ex.printStackTrace();
		} finally {
			//PythonSession.releaseSession(this);
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
		setShouldImpute( Utils.getFlag("im", options) );
		setShouldBinarize( Utils.getFlag("bn", options) );
		setShouldStandardize( Utils.getFlag("sd", options) );
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
		if( getShouldImpute() ) {
			result.add("-im");
		}
		if( getShouldBinarize() ) {
			result.add("-bn");
		}
		if( getShouldStandardize() ) {
			result.add("-sd");
		}
		
		Collections.addAll(result, super.getOptions());
	    return result.toArray(new String[result.size()]);
	}
	
	public static void main(String[] argv) {
		runClassifier(new PyScriptClassifier(), argv);
	}

	@Override
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {
		double[][] dists = new double[insts.numInstances()][insts.numClasses()];
		for(int i = 0; i < insts.numInstances(); i++) {
			dists[i] = distributionForInstance(insts.get(i));
		}
		return dists;
	}

}
