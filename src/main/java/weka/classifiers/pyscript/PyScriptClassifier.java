package weka.classifiers.pyscript;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.BatchPredictor;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import weka.pyscript.Utility;
import weka.python.PythonSession;

/**
 * WEKA class that calls an arbitrary Python script that can take
 * training and testing instances (in the form of Numpy arrays)
 * and return predictions.
 * @author cjb60
 *
 */
public class PyScriptClassifier extends AbstractClassifier implements
	  BatchPredictor, CapabilitiesHandler, TechnicalInformationHandler, OptionHandler {
	
	private static final long serialVersionUID = 2846535265976949760L;
	
	/**
	 * Default values for the parameters.
	 */
	private final String DEFAULT_PYFILE = "";
	private final String DEFAULT_TRAIN_PYFILE_PARAMS = "";
	
	private final boolean DEFAULT_SHOULD_STANDARDIZE = false;
	private final boolean DEFAULT_SHOULD_BINARIZE = false;
	private final boolean DEFAULT_SHOULD_IMPUTE = false;

	private final int DEFAULT_SEED = 0;
	private final String DEFAULT_ARGS_DUMP_FILE = "";
	private final String DEFAULT_PYTHON_COMMAND = "python";
	
	private boolean m_shouldStandardize = DEFAULT_SHOULD_STANDARDIZE;
	private boolean m_shouldBinarize = DEFAULT_SHOULD_BINARIZE;
	private boolean m_shouldImpute = DEFAULT_SHOULD_IMPUTE;

	private String m_argsDumpFile = DEFAULT_ARGS_DUMP_FILE;
	private String m_pythonCommand = DEFAULT_PYTHON_COMMAND;
	
	private Filter m_nominalToBinary = null;
	private Filter m_standardize = null;
	private Filter m_replaceMissing = null;
	
	protected String m_batchPredictSize = "100";
	
	private String m_pickledModel = null;
	
	private transient PythonSession m_session = null;
	
	private String m_modelString = null;
	
	/*
	private int m_numClasses = 0;
	private int m_numAttributes = 0;
	private int m_numInstances = 0;
	private String m_relationName = "";
	private String m_className = null;
	private String[] m_attrNames = null;
	*/
	
	/** The default Python script to execute */
	private String m_pyTrainFile = DEFAULT_PYFILE;
	
	/** If there are any parameters to pass to the training script */
	private String m_customArgs = DEFAULT_TRAIN_PYFILE_PARAMS;
	
	public String getPythonCommand() {
		return m_pythonCommand;
	}
	
	public void setPythonCommand(String s) {
		m_pythonCommand = s;
	}
	
	public String argsDumpFileTipText() {
		return "args dump";
	}
	
	public String getArgsDumpFile() {
		return m_argsDumpFile;
	}
	
	public void setArgsDumpFile(String s) {
		m_argsDumpFile = s;
	}
	
	public String getPythonFile() {
		return m_pyTrainFile;
	}
	
	public void setPythonFile(String pyFile) {
		m_pyTrainFile = pyFile;
	}
	
	public String getCustomArguments() {
		return m_customArgs;
	}
	
	public void setCustomArguments(String pyTrainFileParams) {
		m_customArgs = pyTrainFileParams;
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
	
	public String getModelString() {
		return m_modelString;
	}
	
	/**
	 * Write out the args object to a gzipped pickle
	 * for debugging purposes.
	 * @param trainMode if true, the filename will end in ".train.pkl.gz",
	 * else ".test.pkl.gz"
	 * @throws WekaException
	 */
	public void pickleArgs(boolean trainMode) throws WekaException {
	    // if args dump is set, then save it out to file
	    if( !getArgsDumpFile().equals("") ) {
	    	// see if file exists, if it does then append a number to it
	    	// e.g. if mnist.pkl.gz exists, then do mnist.pkl.gz.1
	    	String currentFilenameTemplate = getArgsDumpFile();
	    	if(trainMode) {
	    		currentFilenameTemplate += ".train";
	    	} else {
	    		currentFilenameTemplate += ".test";
	    	}
	    	
	    	String currentFilename = currentFilenameTemplate;
	    	int idx = 0;
	    	while(true) {
	    		if( new File(currentFilename).exists() ) {
	    			currentFilename = currentFilenameTemplate + "." + idx;
	    			idx += 1;
	    		} else {
	    			break;
	    		}
	    	}
	    	StringBuilder sb = new StringBuilder();
	    	sb.append("import gzip\nimport cPickle as pickle\n");
	    	sb.append("_g = gzip.open('" + currentFilename + "', 'wb')\n");
	    	sb.append("pickle.dump(args, _g, pickle.HIGHEST_PROTOCOL)\n");
	    	sb.append("_g.close()\n");
	    	m_session.executeScript(sb.toString(), getDebug());
	    }
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		try {
		
			// see if the python file exists
			if( ! new File(getPythonFile()).exists() ) {
				throw new FileNotFoundException( getPythonFile() + " doesn't exist!");
			}
			
			m_session = Utility.initPythonSession( this, getPythonCommand(), getDebug() );
			
	    	// now load training and testing class
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + getPythonFile() + "')\n";
	    	List<String> out = m_session.executeScript(driver, getDebug());
		    if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
		    	throw new Exception( "An error happened while trying to load the Python scriptn:\n" + out.get(1) );
		    }    	
	    	
	    	data = Utility.preProcessData(data, getShouldImpute(),
	    			getShouldBinarize(), getShouldStandardize());
		
			
			Utility.pushArgs(data, getCustomArguments(), m_session, true);
		    
		    /*
		     * Ok, push the training data to Python. The variables will be called
		     * X and Y, so let's execute to script to rename these.
		     */
		    m_session.instancesToPythonAsScikitLearn(data, "train", false);
		    m_session.executeScript("args['X_train'] = X\nargs['y_train']=Y\n", getDebug());
		    
		    // build the classifier
		    driver = "model = cls.train(args)";
		    out = m_session.executeScript(driver, getDebug());
		    if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
		    	throw new Exception( "An error happened while executing the train() function:\n" + out.get(1) );
		    }
		    
		    // save model parameters
		    m_pickledModel = m_session.getVariableValueFromPythonAsPickledObject("model", getDebug());
		    
		    // get model description
		    driver = "model_description = cls.describe(args, model)";
		    m_session.executeScript(driver, getDebug());
		    if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
		    	throw new Exception( "An error happened while executing the describe() function:\n" + out.get(1) );
		    }
		    
		    m_modelString = m_session.getVariableValueFromPythonAsPlainString("model_description", getDebug());
	    
		} catch(Exception ex) {
			ex.printStackTrace();
		} finally {
			Utility.closePythonSession(this);
		}

	}

	@Override
	public void setBatchSize(String size) {
		m_batchPredictSize = size;
	}

	@Override
	public String getBatchSize() {
		return m_batchPredictSize;
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public double[] distributionForInstance(Instance inst)
			throws Exception {
		
		Instances insts = new Instances(inst.dataset(), 0);
	    insts.add(inst);
		
		return distributionsForInstances(insts)[0];
		
	}	
	
	@Override
	public void setOptions(String[] options) throws Exception {
		
		String tmp = Utils.getOption("cmd", options);
		if(tmp.length() != 0) {
			setPythonCommand(tmp);
		}
		
		tmp = Utils.getOption("script", options);
		if(tmp.length() != 0) { 
			setPythonFile(tmp);
		}
		
		tmp = Utils.getOption("args", options);
		setCustomArguments(tmp);
		
		setShouldImpute( Utils.getFlag("impute", options) );
		setShouldBinarize( Utils.getFlag("binarize", options) );
		setShouldStandardize( Utils.getFlag("standardize", options) );
		
		tmp = Utils.getOption("df", options);
		setArgsDumpFile(tmp);
		
		tmp = Utils.getOption("-batch", options);
		setBatchSize(tmp);
		
		super.setOptions(options);
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		if( !getPythonCommand().equals("") ) {
			result.add("-cmd");
			result.add( "" + getPythonCommand() );
		}
		if( !getPythonFile().equals("") ) {
			result.add("-script");
			result.add( "" + getPythonFile() );
		}		
		if( !getCustomArguments().equals("") ) {
			result.add("-args");
			result.add( "" + getCustomArguments() );
		}		
		if( getShouldImpute() ) {
			result.add("-impute");
		}
		if( getShouldBinarize() ) {
			result.add("-binarize");
		}
		if( getShouldStandardize() ) {
			result.add("-standardize");
		}		
		if( !getArgsDumpFile().equals("") ) {
			result.add("-df");
			result.add( "" + getArgsDumpFile() );
		}
		result.add("-batch");
		result.add( getBatchSize() );
		
		Collections.addAll(result, super.getOptions());
	    return result.toArray(new String[result.size()]);
	}

	@Override
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {
		
		System.out.format("test = %s\n", insts.numInstances());
		System.out.format("batch size = %s\n", getBatchSize());
		
		double[][] dists = new double[insts.numInstances()][insts.numClasses()];
		
	    m_session = Utility.initPythonSession(this, getPythonCommand(), getDebug());
	    
	    try {
	    	
	    	Utility.preProcessData(insts, getShouldImpute(), getShouldBinarize(), getShouldStandardize());
	        
	        int numClasses = insts.numClasses();
	        
	        // remove the class attribute
	        Remove r = new Remove();
	        r.setAttributeIndices("" + (insts.classIndex() + 1));
	        r.setInputFormat(insts);
	        insts = Filter.useFilter(insts, r);
	        insts.setClassIndex(-1);
		    
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + getPythonFile() + "')\n";
	    	List<String> out = m_session.executeScript(driver, getDebug());
		    if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
		    	throw new Exception( "An error happened while trying to load the Python script:\n" + out.get(1) );
		    }
	        
		    //m_session.executeScript("args = dict()", getDebug());
		    Utility.pushArgs(insts, getCustomArguments(), m_session, false);
		    
		    //pickleArgs(false);
		    
		    /*
		     * Push the test data. These will also be X and Y, so have a
		     * script that renames these to X_test and y_test.
		     */
		    m_session.instancesToPythonAsScikitLearn(insts, "test", false);
		    m_session.executeScript("args['X_test'] = X\n", getDebug());
		    
		    /*
		     * Push the weights of the saved model over.
		     */
		    m_session.setPythonPickledVariableValue("best_weights", m_pickledModel, getDebug());	     
		    
		    driver = "preds = cls.test(args, best_weights)";
		    
		    out = m_session.executeScript(driver, getDebug());
		    if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
		    	throw new Exception( "An error happened while executing the test() function:\n" + out.get(1) );
		    }
		    
			List<Object> preds = 
		    	(List<Object>) m_session.getVariableValueFromPythonAsJson("preds", getDebug());

			for(int y = 0; y < preds.size(); y++) {
		    	Object vector = preds.get(y);
		    	double[] probs = new double[numClasses];
				List<Double> probsForThisInstance = (List<Double>) vector;
		    	for(int x = 0; x < probs.length; x++) {
		    		probs[x] = probsForThisInstance.get(x);
		    	}
		    	dists[y] = probs;
			}
	    	
	    	return dists;
			
	    } catch(Exception ex) {
			ex.printStackTrace();
		} finally {
			Utility.closePythonSession(this);
		}
	    
	    return null;
		
	}
	
	@Override
	public String toString() {
		return m_modelString;
	}
	
	public String globalInfo() {  
		return "Class for calling classifiers that are Python scripts."
		+ "For more information, see\n\n"
		+ getTechnicalInformation().toString();
	}
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.TECHREPORT);
		result.setValue(Field.AUTHOR, "C. Beckham");
		result.setValue(Field.TITLE, "A simple approach to create Python classifiers for WEKA" );
		return result;
	}
	
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>();
		newVector.addElement(
			new Option("\tPython script", "pythonFile", 1, "-fn <filename>")
		);
		newVector.addElement(
			new Option("\tTraining arguments", "trainPythonFileParams", 1, "-xp <string>")
		);
		newVector.addElement(
			new Option("\tTesting arguments", "testPythonFileParams", 1, "-yp <string>")
		);
		newVector.addElement(
			new Option("\tShould we binarise nominal attributes?", "shouldBinarize", 0, "-bn")
		);
		newVector.addElement(
			new Option("\tShould we impute missing values with mean?", "shouldImpute", 0, "-im")
		);
		newVector.addElement(
			new Option("\tShould we standardize the attributes?", "shouldStandardize", 0, "-sd")
		);
		newVector.addElement(
			new Option("\tSet aside 25% of training data as validation data?", "useValidationSet", 0, "-vs")
		);
		newVector.addElement(
			new Option("\tSpecify a filename to dump pickled ARFF to (for debugging purposes)", "argsDumpFile", 1, "-df <filename>")
		);
		newVector.addAll(Collections.list(super.listOptions()));
		return newVector.elements();
	}
	
	public static void main(String[] argv) {
		runClassifier(new PyScriptClassifier(), argv);
	}

}
