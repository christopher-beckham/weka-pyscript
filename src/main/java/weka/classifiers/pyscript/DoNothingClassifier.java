package weka.classifiers.pyscript;

import weka.classifiers.AbstractClassifier;
import weka.core.BatchPredictor;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Classifier that does absolutely nothing.
 * @author cjb60
 */
public class DoNothingClassifier extends AbstractClassifier implements BatchPredictor {

	private static final long serialVersionUID = -2306901982001148346L;
	
	private String m_batchSize = "100";
	
	@Override
	public void buildClassifier(Instances data) throws Exception {}

	@Override
	public void setBatchSize(String size) {
		m_batchSize = size;
	}

	@Override
	public String getBatchSize() {
		return m_batchSize;
	}
	
	@Override
	public double[] distributionForInstance(Instance inst) throws Exception {	
		System.out.println("distributionForInstance");
		Instances insts = new Instances(inst.dataset(), 0);
	    insts.add(inst);
		return distributionsForInstances(insts)[0];
	}	

	@Override
	public double[][] distributionsForInstances(Instances insts) throws Exception {
		System.out.println("batchSize: " + getBatchSize());
		double[][] dist = new double[ insts.numInstances() ][ insts.numClasses() ];
		for(int i = 0; i < insts.numInstances(); i++) {
			dist[i][0] = 1;
		}
		return dist;
	}

}
