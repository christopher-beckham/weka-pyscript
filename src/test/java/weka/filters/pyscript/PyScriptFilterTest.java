package weka.filters.pyscript;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Ignore;
import org.junit.Test;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.pyscript.PyScriptClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PyScriptFilterTest {
	
	/**
	 * See if the standardise filter example behaves in the
	 * exact same way as the one in WEKA.
	 * @throws Exception
	 */
	@Test
	public void testStandardise() throws Exception {
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		PyScriptFilter filter = new PyScriptFilter();
		filter.setPythonFile(new File("scripts/standardise.py"));
		filter.setInputFormat(data);
		
		Instances pyscriptData = Filter.useFilter(data, filter);
		
		Standardize filter2 = new Standardize();
		filter2.setInputFormat(data);
		
		Instances defaultStdData = Filter.useFilter(data, filter2);
		
		// test instances
		for(int x = 0; x < data.numInstances(); x++) {
			assertTrue( pyscriptData.get(x).toString().equals(defaultStdData.get(x).toString()) );
		}
	}
	
	/**
	 * Test to see if regression dataset works. For this we're just
	 * making sure that an exception isn't thrown.
	 */
	@Test
	public void testRegressionDataset() throws Exception {
		DataSource ds = new DataSource("datasets/diabetes_numeric.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex(data.numAttributes() - 1 );

		PyScriptFilter filter = new PyScriptFilter();
		filter.setPythonFile(new File("scripts/standardise.py"));
		filter.setInputFormat(data);
		
		Filter.useFilter(data, filter);
	}
	
	/**
	 * Make sure that the class attribute was modified 
	 * (i.e., ignore class attribute).
	 */
	@Test
	public void testIgnoreClass() throws Exception {
		DataSource ds = new DataSource("datasets/diabetes_numeric.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex(data.numAttributes() - 1 );
		double[] classAttributeValues = data.attributeToDoubleArray(data.classIndex());
		
		PyScriptFilter filter = new PyScriptFilter();
		filter.setPythonFile(new File("scripts/standardise.py"));
		filter.setIgnoreClass(true);
		filter.setInputFormat(data);
		
		Instances filteredData = Filter.useFilter(data, filter);
		
		assert( classAttributeValues[0] != filteredData.attributeToDoubleArray(filteredData.classIndex())[0] );
	}

	/**
	 * Test to see if the script save feature works.
	 * @throws Exception
	 */
	@Test
	public void testSaveScript() throws Exception {
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		PyScriptFilter filter = new PyScriptFilter();
		filter.setPythonFile(new File("scripts/standardise.py"));
		filter.setSaveScript(true);
		filter.determineOutputFormat(data);
		// ok, now change the script
		filter.setPythonFile(new File("not-a-real-file.py"));
		filter.process( data );
	}
	
	/**
	 * Test filtered classifier using PyScriptFilter with
	 * PyScriptClassifier. Just seeing if no exceptions
	 * are thrown here.
	 */
	@Test
	public void testFilteredClassifier() throws Exception {
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		
		FilteredClassifier fs = new FilteredClassifier();
		
		PyScriptClassifier pyScriptClassifier = new PyScriptClassifier();
		pyScriptClassifier.setPythonFile(new File("scripts/scikit-rf.py"));
		pyScriptClassifier.setArguments("num_trees=10;");
		
		PyScriptFilter filter = new PyScriptFilter();
		filter.setPythonFile(new File("scripts/standardise.py"));

		fs.setClassifier(pyScriptClassifier);
		fs.setFilter(filter);
		
		fs.buildClassifier(data);
		fs.distributionsForInstances(data);
	}
	
}
