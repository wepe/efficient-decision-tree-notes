//parse the csv file, get features and label. the format: feature1,feature2,...,label
//first scan, get the feature dimension, dataset size, count of missing value for each feature
//second scan, get each feature's (value,index) and missing value indexes
//if we use ArrayList,only one scanning is needed, but it is of high memory consumption

package histogram_based;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Data{
    //we use -Double.MAX_VALUE to represent missing value
    public static float NULL = -Float.MAX_VALUE;
}

class TrainData extends Data {
    public float[][][] feature_value_index;
    public double[] label;
    public int[][] missing_index;
    public int feature_dim;
    public int dataset_size;
    private ArrayList<Integer> missing_count = new ArrayList<>();
    public float[][] origin_feature;
    private ArrayList<String> cat_features_names;
    public ArrayList<Integer> cat_features_cols = new ArrayList<>();

    //each feature's (value,count)
    public ArrayList<TreeMap<Float,Integer>> feature_value_cnt = new ArrayList<>();
    //feature index and the corresponding upper bounder
    public HashMap<Integer,ArrayList<Float>> numeric_binning = new HashMap<>();
    //feature index and the corresponding categories
    public HashMap<Integer,ArrayList<ArrayList<Float>>> cat_binning = new HashMap<>();


    public TrainData(String file,ArrayList<String> categorical_features){
        this.cat_features_names = categorical_features;
        first_scan(file);
        binning();
        second_scan(file);
    }

    //to obtain: feature_dim, dataset_size,missing_count,cat_features_dim,feature_value_cnt
    private void first_scan(String file){
        try {
            //the header
            BufferedReader br = new BufferedReader(new FileReader(file));
            String header = br.readLine();
            String[] columns = header.split(",");
            feature_dim = columns.length - 1;
            //initialize feature_value_cnt
            for(int i=0;i<feature_dim;i++){
                feature_value_cnt.add(new TreeMap<>());
            }
            //category features' column index
            for(int i=0;i<columns.length;i++){
                if(cat_features_names.contains(columns[i])){
                    cat_features_cols.add(i);
                }
            }

            for(int i=0;i<feature_dim;i++){
                missing_count.add(0);
            }
            //the content
            String line;
            float value;
            TreeMap<Float,Integer> map;
            dataset_size = 0;
            while((line = br.readLine()) != null){
                String[] strs = line.split(",");
                dataset_size += 1;
                for(int i=0;i<feature_dim;i++){
                    if(strs[i].equals("")){
                        missing_count.set(i,missing_count.get(i)+1);
                    }else {
                        value = Float.parseFloat(strs[i]);
                        map = feature_value_cnt.get(i);
                        map.put(value,map.getOrDefault(value,0)+1);
                    }
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /*conduct binning on numeric and category feature
    *　for numeric feature, find the upper bounder of each bin
    * for category feature, find the categories of each bin
    *
    *分箱技巧：使用byte存储数据，区间数255个，因为留了一个表示缺失值．
    * １．数值型特征，先统计每种数值出现个数
    * 如果数值种数小于区间总数，则每种数值一个区间．
    * 否则，按照数值大小排序，然后：
    * 用总样本数/区间数　得到平均每个区间的样本数avg
    * 从小到大遍历数值，累计样本数．
    * 一旦大于avg且小于2avg则这些数值分在一个区间
    * 如果大于2avg，则当前数值之前的累计数值分在一个区间，当前数值也分为一个区间
    * ２．类别型特征，与数值特征相似，不同在于它排序时是按照类别出现个数排序，而不是按类别的值排序
    * */
    private void binning(){
        TreeMap<Float,Integer> map;
        for(int i=0;i<feature_dim;i++){
            map = feature_value_cnt.get(i);

            if(cat_features_cols.contains(i)){
                //it is category feature
                ArrayList<ArrayList<Float>> bins_categories = new ArrayList<>();
                if(map.size()<=255){
                    for(float key:map.keySet()){
                        ArrayList<Float> temp = new ArrayList<>();
                        temp.add(key);
                        bins_categories.add(temp);
                    }
                }else {
                    //sort by value
                    float[][] val_cnt = new float[map.size()][2];
                    int ind = 0;
                    for(float key:map.keySet()){
                        val_cnt[ind][0] = key;
                        val_cnt[ind][1] = map.get(key);
                        ind++;
                    }
                    Arrays.sort(val_cnt, new Comparator<float[]>() {
                        @Override
                        public int compare(float[] a, float[] b) {
                            return Float.compare(a[1],b[1]);
                        }
                    });
                    //binning
                    int total_cnt = 0;
                    for(float key:map.keySet()){
                        total_cnt += map.get(key);
                    }
                    int avg_cnt = total_cnt/255 + 1;

                    int acc_cnt = 0;
                    int last_j = 0;
                    for(int j=0;j<val_cnt.length;j++){
                        acc_cnt += val_cnt[j][1];

                        if(acc_cnt>=avg_cnt && acc_cnt<2*avg_cnt){
                            //put last_j ~ j
                            ArrayList<Float> temp = new ArrayList<>();
                            for(int jj=last_j;jj<=j;jj++){
                                temp.add(val_cnt[jj][0]);
                            }
                            bins_categories.add(temp);

                            acc_cnt = 0;
                            last_j = j+1;
                        }else if(acc_cnt>=2*avg_cnt){
                            //put last_j~(j-1), j
                            ArrayList<Float> temp = new ArrayList<>();
                            temp.add(val_cnt[j][0]);
                            bins_categories.add(temp);

                            if(last_j!=j){
                                ArrayList<Float> temp1 = new ArrayList<>();
                                for(int jj=last_j;jj<j;jj++){
                                    temp1.add(val_cnt[jj][0]);
                                }
                                bins_categories.add(temp1);
                            }

                            acc_cnt = 0;
                            last_j = j+1;
                        }
                    }
                    if(acc_cnt!=0){
                        //put last_j ~ end
                        ArrayList<Float> temp = new ArrayList<>();
                        for(int jj=last_j;jj<val_cnt.length;jj++){
                            temp.add(val_cnt[jj][0]);
                        }
                        bins_categories.add(temp);
                    }
                }

                cat_binning.put(i,bins_categories);

            }else{
                //it is numeric feature
                ArrayList<Float> bins_upper_bounder = new ArrayList<>();
                if(map.size()<=255){
                    for(float key:map.keySet()){
                        bins_upper_bounder.add(key);
                    }
                }else {
                    //sort by key, TreeMap support this
                    //binning
                    int total_cnt = 0;
                    for(float key:map.keySet()){
                        total_cnt += map.get(key);
                    }
                    int avg_cnt = total_cnt/255 + 1;

                    int acc_cnt = 0;
                    float last_key = NULL;
                    for(float key:map.keySet()){
                        acc_cnt += map.get(key);
                        if(acc_cnt>=avg_cnt && acc_cnt<2*avg_cnt){
                            bins_upper_bounder.add(key);
                            acc_cnt = 0;
                        }else if(acc_cnt>=2*avg_cnt){
                            if(last_key!=NULL && !bins_upper_bounder.contains(last_key)){
                                bins_upper_bounder.add(last_key);
                            }
                            bins_upper_bounder.add(key);
                            acc_cnt = 0;
                        }
                        last_key = key;
                    }
                    if(acc_cnt!=0){
                        bins_upper_bounder.add(last_key);
                    }

                }

                numeric_binning.put(i,bins_upper_bounder);
            }
        }
        feature_value_cnt = null;
    }

    //存在问题：即便特征的value分箱了，可以用byte存储，但是对应的index还是得用int，也就是不能用整个feature_value_index了，要分成两个数组，代码得重构


    //to obtain:feature_value_index,label,missing_index,origin_feature
    private void second_scan(String file){
        label = new double[dataset_size];
        missing_index = new int[feature_dim][];
        feature_value_index = new float[feature_dim][][];

        for(int i=0;i<feature_dim;i++){
            int cnt = missing_count.get(i);
            missing_index[i] = new int[cnt];
            feature_value_index[i] = new float[dataset_size-cnt][2];
        }

        origin_feature = new float[dataset_size][feature_dim];

        try{
            BufferedReader br = new BufferedReader(new FileReader(file));
            br.readLine();

            int[] cur_index = new int[feature_dim];
            int[] cur_missing_index = new int[feature_dim];
            Arrays.fill(cur_index,0);
            Arrays.fill(cur_missing_index,0);

            for(int row=0;row<dataset_size;row++){
                String[] strs = br.readLine().split(",");
                label[row] = Float.parseFloat(strs[strs.length-1]);

                for(int col=0;col<feature_dim;col++){
                    if(strs[col].equals("")){
                        missing_index[col][cur_missing_index[col]] = row;
                        cur_missing_index[col] += 1;
                        origin_feature[row][col] = Data.NULL;
                    }else{
                        feature_value_index[col][cur_index[col]][0] = Float.parseFloat(strs[col]);
                        feature_value_index[col][cur_index[col]][1] = row;
                        cur_index[col] += 1;
                        origin_feature[row][col] = Float.parseFloat(strs[col]);
                    }
                }
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }
}


class ValidationData extends Data {
    public int feature_dim;
    public int dataset_size;
    public float[][] origin_feature;
    public double[] label;

    public ValidationData(String file){
        first_scan(file);
        second_scan(file);
    }

    private void first_scan(String file){
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String header = br.readLine();
            feature_dim = header.split(",").length - 1;

            String line;
            dataset_size = 0;
            while((line = br.readLine()) != null){
                dataset_size += 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void second_scan(String file){
        label = new double[dataset_size];
        origin_feature = new float[dataset_size][feature_dim];

        try{
            BufferedReader br = new BufferedReader(new FileReader(file));
            br.readLine();
            for(int row=0;row<dataset_size;row++){
                String[] strs = br.readLine().split(",");
                label[row] = Float.parseFloat(strs[strs.length-1]);
                for(int col=0;col<feature_dim;col++){
                    if(strs[col].equals("")){
                        origin_feature[row][col] = Data.NULL;
                    }else{
                        origin_feature[row][col] = Float.parseFloat(strs[col]);
                    }
                }
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }
}


class TestData extends Data {
    public int feature_dim;
    public int dataset_size;
    public float[][] origin_feature;

    public TestData(String file){
        first_scan(file);
        second_scan(file);
    }

    private void first_scan(String file){
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String header = br.readLine();
            feature_dim = header.split(",").length;

            String line;
            dataset_size = 0;
            while((line = br.readLine()) != null){
                dataset_size += 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void second_scan(String file){
        origin_feature = new float[dataset_size][feature_dim];

        try{
            BufferedReader br = new BufferedReader(new FileReader(file));
            br.readLine();
            for(int row=0;row<dataset_size;row++){
                String[] strs = br.readLine().split(",");
                for(int col=0;col<feature_dim;col++){
                    if(strs[col].equals("")){
                        origin_feature[row][col] = Data.NULL;
                    }else{
                        origin_feature[row][col] = Float.parseFloat(strs[col]);
                    }
                }
            }
        }catch (IOException e){
            e.printStackTrace();
        }
    }

}
