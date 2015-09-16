package weka.pyscript;

import java.io.File;
import java.io.FileNotFoundException;
import weka.classifiers.pyscript.PyScriptClassifier;
import weka.core.CommandlineRunnable;
import weka.core.SerializationHelper;
import weka.core.Utils;

/**
 * Allows a PyScriptClassifier model to be deserialised, have its
 * Python script path changed, and reserialised.
 * @author cjb60
 */
public class ChangeScriptPath implements CommandlineRunnable {
	
	private static final String USAGE = "Usage: -i <input model> -o <output model> -script <new script path>";
	
	private String m_inputModel = null;
	private String m_outputModel = null;
	private String m_scriptPath = null;
	
	public void setInputModel(String s) {
		m_inputModel = s;
	}
	
	public String getInputModel() {
		return m_inputModel;
	}
	
	public void setOutputModel(String s) {
		m_outputModel = s;
	}
	
	public String getOutputModel() {
		return m_outputModel;
	}
	
	public void setScriptPath(String s) {
		m_scriptPath = s;
	}
	
	public String getScriptPath() {
		return m_scriptPath;
	}
	
	public void setOptions(String[] options) {
		try {
			setInputModel( Utils.getOption("i", options) );
			setOutputModel( Utils.getOption("o", options) );
			setScriptPath( Utils.getOption("script", options) );
		} catch(Exception ex) {
			System.err.println(USAGE);
			ex.printStackTrace();
		}
	}

	public void convert() throws Exception {
		
		if( ! new File( getInputModel() ).exists() ) {
			throw new FileNotFoundException( "Input model " + getInputModel() + " does not exist!");
		}
		
		if( ! new File( getScriptPath() ).exists() ) {
			throw new FileNotFoundException( "Python script " + getScriptPath() + " does not exist!");
		}
		
		PyScriptClassifier cls = (PyScriptClassifier) SerializationHelper.read( getInputModel() );
		
		System.err.println( cls.getModelString() + "\nCurrent script path: " +
				cls.getPythonFile().getAbsolutePath() );		
		
		System.err.println("Changing script path to: " + getScriptPath() );
		
		cls.setPythonFile( new File( getScriptPath() ) );
		
		SerializationHelper.write( getOutputModel(), cls);
		
	}

	@Override
	public void run(Object toRun, String[] options) {
		try {
			ChangeScriptPath csp = (ChangeScriptPath) toRun;
			csp.setOptions(options);
			csp.convert();
		} catch(Exception ex) {
			System.err.println(USAGE);
			ex.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		new ChangeScriptPath().run(new ChangeScriptPath(), args);
	}
	
}
