package weka.core;

import weka.core.CommandlineRunnable;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.pyscript.Utility;
import weka.python.PythonSession;

/**
 * Convert an .arff file to a pkl.gz for the purposes of
 * testing Python classifiers independently of WEKA.
 * @author cjb60
 */
public class ArffToPickle implements CommandlineRunnable {
	
	private PythonSession m_session = null;
	private boolean m_debug = false;
	
	private String m_filename = null;
	private String m_dest = null;
	
	private String m_classIndex = "last";
	
	public void setFilename(String s) {
		m_filename = s;
	}
	
	public void setDest(String s) {
		m_dest = s;
	}
	
	public boolean getDebug() {
		return m_debug;
	}
	
	public void setDebug(boolean b) {
		m_debug = b;
	}
	
	public String getClassIndex() {
		return m_classIndex;
	}
	
	public void setClassIndex(String c) {
		m_classIndex = c;
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
	
	public void convert() {
		try {
			initPythonSession();
			if(m_session == null) {
				m_session = PythonSession.acquireSession(this);
			}			
			DataSource ds = new DataSource(m_filename);
			Instances instances = ds.getDataSet();
			
			if(m_classIndex.equals("first")) {
				instances.setClassIndex(0);
			} else if (m_classIndex.equals("last") ) {
				instances.setClassIndex( instances.numAttributes() - 1 );
			} else {
				instances.setClassIndex( Integer.parseInt(m_classIndex) );
			}
			
			Utility.pushArgs(instances, "", m_session, m_debug);
			
		    m_session.instancesToPythonAsScikitLearn(instances, "train", false);
		    m_session.executeScript("args['X_train'] = X\nargs['y_train'] = Y\n", getDebug());
			
	    	StringBuilder sb = new StringBuilder();
	    	sb.append("import gzip\nimport cPickle as pickle\n");
	    	sb.append("_g = gzip.open('" + m_dest.replace("'","\\'") + "', 'wb')\n");
	    	sb.append("pickle.dump(args, _g, pickle.HIGHEST_PROTOCOL)\n");
	    	sb.append("_g.close()\n");
	    	m_session.executeScript(sb.toString(), m_debug);			
		} catch(Exception ex) {
			ex.printStackTrace();
		} finally {
			closePythonSession();
		}
	}

	@Override
	public void run(Object toRun, String[] options)
			throws IllegalArgumentException {
		if( options.length != 6 && options.length != 7 ) {
			throw new IllegalArgumentException("Usage: -i <arff file> -o <destination file> -c <class index> [-debug]");
		}
		ArffToPickle arffToPickle = (ArffToPickle) toRun;
		for(int i = 0; i < options.length; i += 2) {
			if( options[i].equals("-i")) {
				arffToPickle.setFilename(options[i+1]);
			} else if( options[i].equals("-o")) {
				arffToPickle.setDest(options[i+1]);
			} else if( options[i].equals("-c") ) {
				arffToPickle.setClassIndex( options[i+1] );
			} else if( options[i].equals("-debug")) {
				arffToPickle.setDebug(true);
			} else {
				throw new IllegalArgumentException("Unknown flag: " + options[i] +
					"\n" + "Usage: -i <arff file> -o <destination file> -c <class index> [-debug]");
			}
		}
		arffToPickle.convert();
	}
}
