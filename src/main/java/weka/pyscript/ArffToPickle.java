package weka.pyscript;

import weka.core.CommandlineRunnable;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.python.PythonSession;

class ArffToPickle implements CommandlineRunnable {
	
	private PythonSession m_session = null;
	private boolean m_debug = true;
	
	private String m_filename = null;
	private String m_dest = null;
	
	public void setFilename(String s) {
		m_filename = s;
	}
	
	public void setDest(String s) {
		m_dest = s;
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
	
	public void test() {
		try {
			initPythonSession();
			if(m_session == null) {
				m_session = PythonSession.acquireSession(this);
			}			
			DataSource ds = new DataSource(m_filename);
			Instances instances = ds.getDataSet();
			m_session.executeScript("args = dict()", m_debug);
			Utility.pushArgs(instances, "", m_session, m_debug);
			
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
		((ArffToPickle)toRun).setFilename(options[0]);
		((ArffToPickle)toRun).setDest(options[1]);
		((ArffToPickle)toRun).test();
	}
	
	//public static void main(String[] args) {
	//	ArffToPickle x = new ArffToPickle();
	//	x.run(x, args);
	//}
}
