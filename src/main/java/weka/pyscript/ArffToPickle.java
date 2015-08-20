package weka.pyscript;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.python.PythonSession;

class ArffToPickleSession {
	
	private String m_filename = null;
	private PythonSession m_session = null;
	private boolean m_debug = false;
	
	public ArffToPickleSession(String filename) {
		m_filename = filename;
	}
	
	public void initPythonSession() throws Exception {
	    if (!PythonSession.pythonAvailable()) {
	        if (!PythonSession.initSession( "python", true)) {
	          String envEvalResults = PythonSession.getPythonEnvCheckResults();
	          throw new Exception("Was unable to start python environment: "
	            + envEvalResults);
	        }
	    } 
	}
	
	public void closePythonSession() {
		PythonSession.releaseSession(this);
	}
	
	public void test() throws Exception {
		if(m_session == null) {
			m_session = PythonSession.acquireSession(this);
		}
		DataSource ds = new DataSource(m_filename);
		Instances instances = ds.getDataSet();
		m_session.instancesToPython(instances, "df", true);
	}
	
}

/**
 * Convert an .arff file into a .pkl.gz to be able
 * to test Python scripts on .arff files without needing
 * WEKA
 * @author cjb60
 *
 */
public class ArffToPickle {
	
	public static void main(String[] args) throws Exception {
		
		ArffToPickleSession session = new 
				ArffToPickleSession("/Users/cjb60/Desktop/weka/datasets/UCI/iris.arff");
		
		//session.initPythonSession();
		
		//session.closePythonSession();
		
	}

}
