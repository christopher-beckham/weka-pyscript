package weka.filters.pyscript;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.List;

import weka.classifiers.pyscript.PyScriptClassifier;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.SimpleBatchFilter;
import weka.pyscript.Utility;
import weka.python.PythonSession;

/**
 * WEKA class that calls an arbitrary Python script that can
 * transform the data (i.e. act as a WEKA filter).
 * @author cjb60
 *
 */
public class PyScriptFilter extends SimpleBatchFilter {

	private static final long serialVersionUID = -6127927234772532696L;
	
	private transient PythonSession m_session = null;
	
	private final String DEFAULT_PYTHON_COMMAND = "python";
	private final File DEFAULT_PYFILE = new File( System.getProperty("user.dir") );
	private final String DEFAULT_TRAIN_PYFILE_PARAMS = "";
	
	private String m_pythonCommand = DEFAULT_PYTHON_COMMAND;
	
	/** The default Python script to execute */
	private File m_pyTrainFile = DEFAULT_PYFILE;
	
	/** If there are any parameters to pass to the training script */
	private String m_customArgs = DEFAULT_TRAIN_PYFILE_PARAMS;
	
	private String m_argsScript = null;
	
	private String m_pickledModel = null;
	private String m_modelString = null;
	
	public String getArguments() {
		return m_customArgs;
	}
	
	public void setArguments(String pyTrainFileParams) {
		m_customArgs = pyTrainFileParams;
	}

	@OptionMetadata(
		displayName = "pythonCommand",
		description = "Python executable command", commandLineParamName = "cmd",
		commandLineParamSynopsis = "-cmd <python executable>", displayOrder = 4
	)
	public String getPythonCommand() {
		return m_pythonCommand;
	}

	public void setPythonCommand(String s) {
		m_pythonCommand = s;
	}
	
	@OptionMetadata(
		displayName = "pythonFile",
		description = "Path to Python script", commandLineParamName = "script",
		commandLineParamSynopsis = "-script <path to script>", displayOrder = 4
	)
	public File getPythonFile() {
		return m_pyTrainFile;
	}

	public void setPythonFile(File pyFile) {
		m_pyTrainFile = pyFile;
	}

	@Override
	public String globalInfo() {
		return null;
	}
	
	public boolean getPrintStdOut() {
		return true;
	}
	
	private void executeScript(String driver, String stdErrMessage) throws Exception {
		List<String> out = m_session.executeScript(driver, getDebug());
		if( stdErrMessage != null) {
			if(out.get(1).contains(Utility.TRACEBACK_MSG)) {
				throw new Exception(stdErrMessage + "\n" + out.get(1));
			}
		}
		if( getPrintStdOut() && !out.get(0).equals("") ) {
			System.err.println( "Standard out:\n" + out.get(0) );
		}
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		
		try {
			
			Instances data = inputFormat;
			
			// first train the filter
			m_session = Utility.initPythonSession( this, getPythonCommand(), getDebug() );
			
			// set the working directory of the python vm to that of the script
			String parentDir = getPythonFile().getAbsoluteFile().getParent();
			String scriptName = getPythonFile().getName();
			if(parentDir != null) {
				String driver = "import os\nos.chdir('" + parentDir + "')\n";
				driver += "import sys\nsys.path.append('" + parentDir + "')\n";
				executeScript(driver, "An error happened while trying to change the working directory:");
			}
			
	    	// now load training and testing class
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + scriptName + "')\n";
	    	executeScript(driver, "An error happened while trying to load the Python script:");
	    	
			m_argsScript = Utility.createArgsScript(data, getArguments(), m_session, getDebug());		
			executeScript(m_argsScript, "An error happened while trying to create the args variable:");
	    	
		    /*
		     * Ok, push the training data to Python. The variables will be called
		     * X and Y, so let's execute to script to rename these.
		     */
		    m_session.instancesToPythonAsScikitLearn(data, "train", false);
		    m_session.executeScript("args['X_train'] = X\nargs['y_train']=Y\n", getDebug());
		    
		    // build the classifier
		    driver = "model = cls.train(args)";	    
		    executeScript(driver, "An error happened while executing the train() function:");
		    
		    // save model parameters
		    m_pickledModel = m_session.getVariableValueFromPythonAsPickledObject("model", getDebug());
	
		    // get model description
		    //driver = "model_description = cls.describe(args, model)";
		    //executeScript(driver, "An error happened while executing the describe() function:");
		    //m_modelString = m_session.getVariableValueFromPythonAsPlainString("model_description", getDebug());
		    
		    //System.out.println(m_modelString);
		    
		    // ok now filter
		    m_session.executeScript("args['X'] = args['X_train']\nargs['y'] = args['y_train']\n", getDebug());
		    driver = "arff = cls.filter(args, model)";
		    executeScript(driver, "An error happened while executing the filter() function:");
		    
		    String arff = m_session.getVariableValueFromPythonAsPlainString("arff", getDebug());
		    //oSystem.out.println(arff);
		    
		    DataSource ds = new DataSource( new ByteArrayInputStream(arff.getBytes("UTF-8") ) );
		    Instances transformed = ds.getDataSet();
		    
		    // assumption we make for now
		    transformed.setClassIndex( transformed.numAttributes() - 1);
		    
		    //System.out.println(transformed);
			
		    return transformed;
		    
		} catch(Exception ex) {
			ex.printStackTrace();
			throw ex;
		} finally {
			Utility.closePythonSession(this);
		}
		
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		
		try {
			m_session = Utility.initPythonSession( this, getPythonCommand(), getDebug() );
			
			Instances data = instances;
			
			String parentDir = getPythonFile().getAbsoluteFile().getParent();
			String scriptName = getPythonFile().getName();
			if(parentDir != null) {
				String driver = "import os\nos.chdir('" + parentDir + "')\n";
				driver += "import sys\nsys.path.append('" + parentDir + "')\n";
				executeScript(driver, "An error happened while trying to change the working directory:");
			}
			
	    	String driver = "import imp\n"
	    			+ "cls = imp.load_source('cls','" + scriptName + "')\n";
	    	executeScript(driver, "An error happened while trying to load the Python script:");
	    	executeScript(m_argsScript, "An error happened while trying to create the args variable:" );
	    	
		    m_session.instancesToPythonAsScikitLearn(data, "test", false);
		    m_session.executeScript("args['X'] = X\nargs['y'] = Y", getDebug());    	
			
		    m_session.setPythonPickledVariableValue("model", m_pickledModel, getDebug());
		    
		    driver = "arff = cls.filter(args, model)";
		    executeScript(driver, "An error happened while executing the filter() function:");
		    
		    String arff = m_session.getVariableValueFromPythonAsPlainString("arff", getDebug());
		    DataSource ds = new DataSource( new ByteArrayInputStream(arff.getBytes("UTF-8") ) );
		    Instances transformed = ds.getDataSet();
		    
		    return transformed;
	    
		} catch(Exception ex) {
			ex.printStackTrace();
			throw ex;
		} finally {
			Utility.closePythonSession(this);
		}
		
	}
	
	@Override
	public boolean allowAccessToFullInputFormat() {
		return true;
	}
	
	public static void main(String[] argv) {
		runFilter(new PyScriptFilter(), argv);
	}

}
