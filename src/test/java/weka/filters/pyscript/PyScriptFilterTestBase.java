package weka.filters.pyscript;

import java.io.File;

import weka.core.Instances;
import weka.core.TestInstances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.AbstractFilterTest;
import weka.filters.Filter;

public class PyScriptFilterTestBase extends AbstractFilterTest {

	public PyScriptFilterTestBase(String name) {
		super(name);
	}
	
	@Override
	protected void setUp() throws Exception {
		super.setUp();
		DataSource ds = new DataSource("tests/filter-test.arff");
		m_Instances = ds.getDataSet();
		//m_Instances.setClassIndex( m_Instances.numAttributes() - 1);
	}

	@Override
	public Filter getFilter() {
		PyScriptFilter f = new PyScriptFilter();
		f.setPythonFile( new File("tests/do-nothing-filter.py") );
		f.setSaveScript(true);
		return f;
	}
	
	@Override
	protected Instances getFilteredClassifierData() throws Exception {
		
		TestInstances test = TestInstances.forCapabilities(
				m_FilteredClassifier.getClassifier().getCapabilities());
		test.setNumDate(0);
		test.setClassIndex(TestInstances.CLASS_IS_LAST);

		Instances result = test.generate();
		System.out.println(result);

		return result;
	}
	

}
