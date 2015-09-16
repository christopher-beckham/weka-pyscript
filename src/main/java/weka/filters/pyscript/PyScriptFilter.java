package weka.filters.pyscript;

import weka.core.Instances;
import weka.filters.SimpleBatchFilter;

/**
 * WEKA class that calls an arbitrary Python script that can
 * transform the data (i.e. act as a WEKA filter).
 * @author cjb60
 *
 */
public class PyScriptFilter extends SimpleBatchFilter {

	private static final long serialVersionUID = -6127927234772532696L;

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public boolean allowAccessToFullInputFormat() {
		return true;
	}

}
