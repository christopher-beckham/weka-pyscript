package weka.classifiers.pyscript;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
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
import weka.python.PythonSession;

/**
 * WEKA class that calls an arbitrary Python script that can take
 * training and testing instances (in the form of Numpy arrays)
 * and return predictions.
 * @author cjb60
 *
 */
public class PyScriptClassifier extends AbstractClassifier implements
	  BatchPredictor, CapabilitiesHandler, Randomizable, TechnicalInformationHandler, OptionHandler {
	
	private static final long serialVersionUID = 2846535265976949760L;
	
	/**
	 * Default values for the parameters.
	 */
	private final String DEFAULT_PYFILE = "";
	private final String DEFAULT_TRAIN_PYFILE_PARAMS = 
			"";
	private final String DEFAULT_TEST_PYFILE_PARAMS = DEFAULT_TRAIN_PYFILE_PARAMS;
	
	private final boolean DEFAULT_SHOULD_STANDARDIZE = false;
	private final boolean DEFAULT_SHOULD_BINARIZE = false;
	private final boolean DEFAULT_SHOULD_IMPUTE = false;
	private final boolean DEFAULT_USE_VALIDATION_SET = false;
	private final int DEFAULT_SEED = 0;
	private final String DEFAULT_ARGS_DUMP_FILE = "";
	private final String DEFAULT_PYTHON_COMMAND = "python";
	
	private boolean m_shouldStandardize = DEFAULT_SHOULD_STANDARDIZE;
	private boolean m_shouldBinarize = DEFAULT_SHOULD_BINARIZE;
	private boolean m_shouldImpute = DEFAULT_SHOULD_IMPUTE;
	private boolean m_useValidationSet = DEFAULT_USE_VALIDATION_SET;
	private int m_seed = DEFAULT_SEED;
	private String m_argsDumpFile = DEFAULT_ARGS_DUMP_FILE;
	private String m_pythonCommand = DEFAULT_PYTHON_COMMAND;
	
	private Filter m_nominalToBinary = null;
	private Filter m_standardize = null;
	private Filter m_replaceMissing = null;
	private HashMap<String, ArrayList<String>> m_attrEnums = null;
	
	private String m_pickledModel = null;
	
	private transient PythonSession m_session = null;
	
	private String m_modelString = null;
	
	private int m_numClasses = 0;
	private int m_numAttributes = 0;
	private int m_numInstances = 0;
	private String m_relationName = "";
	private String m_className = null;
	private String[] m_attrNames = null;
	
	/** The default Python script to execute */
	private String m_pyTrainFile = DEFAULT_PYFILE;
	
	/** If there are any parameters to pass to the training script */
	private String m_pyTrainFileParams = DEFAULT_TRAIN_PYFILE_PARAMS;
	
	/** If there are any parameters to pass to the testing script */
	private String m_pyTestFileParams = DEFAULT_TEST_PYFILE_PARAMS;
	
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
	
	private void pushArgs(PythonSession session, boolean trainMode) throws Exception {
		
		// pass general information related to the training data
	    session.executeScript("args['num_classes'] = " + m_numClasses, getDebug());
	    session.executeScript("args['num_attributes'] = " + m_numAttributes, getDebug());
	    session.executeScript("args['num_instances'] = " + m_numInstances, getDebug());
	    session.executeScript("args['relation_name'] = " +
	    		"'" + m_relationName.replace("'", "") + "'", getDebug());
	    
	    // pass attribute information
	    StringBuilder attrNames = new StringBuilder("args['attributes'] = [");
	    for(int i = 0; i < m_numAttributes; i++) {
	    	String attrName = m_attrNames[i];
	    	attrName = attrName.replace("'", "").replace("\"", "");
	    	attrNames.append( "'" + attrName + "'" );
	    	if(i != m_numAttributes-1) {
	    		attrNames.append(",");
	    	}
	    }
	    attrNames.append("]");
	    session.executeScript( attrNames.toString(), getDebug());
	    
	    // pass attribute enums
	    StringBuilder attrValues = new StringBuilder("args['attr_values'] = dict()\n");
	    for(String key : m_attrEnums.keySet()) {
	    	StringBuilder vector = new StringBuilder();
	    	vector.append("[");
	    	ArrayList<String> vals = m_attrEnums.get(key);
	    	for(String val : vals) {
	    		vector.append( "'" + val + "'" + "," );
	    	}
	    	vector.append("]");
	    	attrValues.append("args['attr_values']['" + 
	    		key.replace("'", "\\'").replace("\n", "\\n") + "'] = " + vector.toString() );
	    }
	    session.executeScript(attrValues.toString(), getDebug());
	    
	    // pass class name
	    String classAttr = m_className.replace("'", "").replace("\"", "");
	    session.executeScript( "args['class'] = '" + classAttr.replace("'", "") + "'", getDebug());    
	    
	    // pass custom parameters from -xp or -yp
	    String customParams = null;
	    if(trainMode) {
	    	customParams = getTrainPythonFileParams();
	    } else {
	    	customParams = getTestPythonFileParams();
	    }
	    if( !customParams.equals("") ) {
		    String[] extraParams = customParams.split(",");
		    for(String param : extraParams) {
		    	String[] paramSplit = param.split("=");
		    	session.executeScript("args[" + paramSplit[0] + "] = " + paramSplit[1], getDebug());
		    }
	    }
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
	public void buildClassifier(Instances data) {
		
		try {
		
			// see if the python file exists
			if( ! new File(getPythonFile()).exists() ) {
				throw new FileNotFoundException( getPythonFile() + " doesn't exist!");
			}
			
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
			
			Instances validData = null;
		    
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
			    stratifiedFolds.setOptions(new String[] {"-S", ""+getSeed(),  "-V",  "-N","4",  "-F","1" } );
			    data = Filter.useFilter(data, stratifiedFolds);
			    // this time, don't invert the selection
			    stratifiedFolds.setOptions( new String[] {"-S",""+getSeed(),  "-N","4",  "-F","1" } );
			    validData = Filter.useFilter(data, stratifiedFolds);
			} else {
				//;
			}
		    
		    m_session.executeScript("args = dict()", getDebug());
		    
		    /*
		     * Ok, push the training data to Python. The variables will be called
		     * X and Y, so let's execute to script to rename these.
		     */
		    m_session.instancesToPythonAsScikitLearn(data, "train", false);
		    m_session.executeScript("args['X_train'] = X\nargs['y_train']=Y\n", getDebug());
		    
		    /*
		     * Push the validation data.
		     */
		    if( getUseValidationSet() ) {
		    	m_session.instancesToPythonAsScikitLearn(validData, "valid", false);
		    	m_session.executeScript("args['X_valid'] = X\nargs['y_valid']=Y\n", getDebug());
		    }
		    
		    //System.out.format("train, valid = %s, %s\n",
		    //		m_trainingData.numInstances(), m_validData.numInstances());
		    
		    m_numClasses = data.numClasses();
		    m_numAttributes = data.numAttributes() - 1;
		    m_numInstances = data.numInstances();
		    m_relationName = data.relationName();
		    m_className = data.classAttribute().name();
		    m_attrNames = new String[ data.numAttributes() - 1 ];
		    m_attrEnums = new HashMap<String, ArrayList<String> >();
		    for(int i = 0; i < data.numAttributes()-1; i++) {
		    	m_attrNames[i] = data.attribute(i).name();
		    	
		    	if( data.attribute(i).isNominal() || data.attribute(i).isString() ) {
			    	Enumeration<Object> en = data.attribute(i).enumerateValues();
			    	ArrayList<String> strs = new ArrayList<String>(data.attribute(i).numValues());
			    	while(en.hasMoreElements()) {
			    		strs.add( (String) en.nextElement() );
			    	}    	
			    	m_attrEnums.put(m_attrNames[i], strs);
		    	}
		    }
	
		    pushArgs(m_session, true);
		    
		    pickleArgs(true);
		    
		    //System.out.println("Number of classes: " + m_numClasses);
		    //System.out.println("Number of attributes: " + m_numAttributes);
		    //System.out.println("Number of instances: " + m_numInstances);
		    
		    // build the classifier
		    String driver = "best_weights = cls.train(args)";
		    List<String> trainOutAndErr = m_session.executeScript(driver, getDebug());
		    
		    // save model parameters
		    m_pickledModel = m_session.getVariableValueFromPythonAsPickledObject("best_weights", getDebug());
		    
		    // get model description
		    driver = "model_desc = cls.describe(args, best_weights)";
		    m_session.executeScript(driver, getDebug());
		    m_modelString = m_session.getVariableValueFromPythonAsPlainString("model_desc", getDebug());
		    //System.out.println("Model string:" + m_modelString);
		    
		    //PythonSession.releaseSession(this);
	    
		} catch(Exception ex) {
			ex.printStackTrace();
		} finally {
			closePythonSession();
		}

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
	
	private void initPythonSession() throws Exception {
	    if (!PythonSession.pythonAvailable()) {
	        // try initializing
	        if (!PythonSession.initSession( getPythonCommand(), getDebug())) {
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
    		
    		//System.err.println("This should only run once per x-val");
    	
	    	// now load training and testing class
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + getPythonFile() + "')\n";
	    	m_session.executeScript(driver, getDebug());
    	}
	    
	}
	
	private void closePythonSession() {
		PythonSession.releaseSession(this);
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public double[] distributionForInstance(Instance inst)
			throws Exception {
		
		System.out.println("distributionForInstance");
		
		Instances insts = new Instances(inst.dataset());
		
		return distributionsForInstances(insts)[0];
		
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		
		String tmp = Utils.getOption("pc", options);
		if(tmp.length() != 0) {
			setPythonCommand(tmp);
		}
		
		tmp = Utils.getOption("fn", options);
		if(tmp.length() != 0) { 
			setPythonFile(tmp);
		}
		
		tmp = Utils.getOption("xp", options);
		setTrainPythonFileParams(tmp);

		tmp = Utils.getOption("yp", options);
		setTestPythonFileParams(tmp);
		
		setShouldImpute( Utils.getFlag("im", options) );
		setShouldBinarize( Utils.getFlag("bn", options) );
		setShouldStandardize( Utils.getFlag("sd", options) );
		
		setUseValidationSet( Utils.getFlag("vs", options) );
		
		tmp = Utils.getOption("df", options);
		setArgsDumpFile(tmp);
		
		super.setOptions(options);
	}
	
	@Override
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();
		if( !getPythonCommand().equals("") ) {
			result.add("-pc");
			result.add( "" + getPythonCommand() );
		}
		if( !getPythonFile().equals("") ) {
			result.add("-fn");
			result.add( "" + getPythonFile() );
		}		
		if( !getTrainPythonFileParams().equals("") ) {
			result.add("-xp");
			result.add( "" + getTrainPythonFileParams() );
		}	
		if( !getTestPythonFileParams().equals("") ) {
			result.add("-yp");
			result.add( "" + getTestPythonFileParams() );
		}		
		if( getShouldImpute() ) {
			result.add("-im");
		}
		if( getShouldBinarize() ) {
			result.add("-bn");
		}
		if( getShouldStandardize() ) {
			result.add("-sd");
		}
		if( getUseValidationSet() ) {
			result.add("-vs");
		}		
		if( !getArgsDumpFile().equals("") ) {
			result.add("-df");
			result.add( "" + getArgsDumpFile() );
		}		
		Collections.addAll(result, super.getOptions());
	    return result.toArray(new String[result.size()]);
	}

	@Override
	public double[][] distributionsForInstances(Instances insts)
			throws Exception {
		
		double[][] dists = new double[insts.numInstances()][insts.numClasses()];
		
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
		    
		    m_session.executeScript("args = dict()", getDebug());
		    pushArgs(m_session, false);
		    
		    pickleArgs(false);
		    
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
		    
		    System.out.format("test = %s\n", insts.numInstances());	    
		    
		    String driver = "preds = cls.test(args, best_weights)";
		    
		    m_session.executeScript(driver, getDebug());
		    
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
			closePythonSession();
		}
	    
	    return null;
		
	}
	
	@Override
	public String toString() {
		return m_modelString;
	}

	@Override
	public void setSeed(int seed) {
		m_seed = seed;
	}

	@Override
	public int getSeed() {
		return m_seed;
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
