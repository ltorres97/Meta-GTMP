import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


df = pd.DataFrame({'Metric': ['Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp'],\
                  'GIN':        np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9956, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9956, 1.0, 1.0]) ,
                  'GCN':        np.asarray([0.9646, 0.854, 0.92, 0.9646, 0.9646, 0.9735, 0.9469, 0.9292, 0.9381, 0.9336, 0.9735, 0.9333, 0.8673, 0.9469, 0.8222, 0.8451, 0.9558, 0.8894, 0.6947, 0.8761, 0.8929, 0.9467, 0.9822, 0.9513, 0.9689, 0.9735, 0.9644, 0.8982, 0.9867, 0.9425]) ,
                  'GraphSAGE':  np.asarray([0.9735, 0.9823, 0.8711, 0.9735, 0.9823, 0.9867, 0.9779, 0.9558, 0.9779, 0.969, 0.9823, 0.8622, 0.969, 0.9823, 0.9822, 0.9779, 0.9779, 0.9779, 0.9779, 0.9646, 0.9732, 0.9778, 0.9822, 0.9823, 0.9867, 0.9867, 0.9822, 0.9646, 0.9867, 0.9823]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9867, 0.9867, 0.9778, 0.9469, 0.9867, 0.9867, 0.9867, 0.9867, 0.9867, 0.9735, 0.9867, 0.9689, 0.9735, 0.9867, 0.9867, 0.9646, 0.9867, 0.9867, 0.9779, 0.9867, 0.9688, 0.9867, 0.9867, 0.9912, 0.9867, 0.9867, 0.9689, 0.9823, 0.9867, 0.9912])
                  })


df = df[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df2 = pd.DataFrame({'Metric': ['Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn'],\
                  'GIN':        np.asarray([0.8403, 0.8329, 0.8575, 0.8646, 0.8426, 0.8403, 0.8154, 0.8546, 0.8413, 0.8853, 0.8381, 0.8553, 0.8497, 0.8226, 0.8498, 0.8591, 0.8542, 0.8423, 0.8559, 0.8348, 0.8579, 0.8284, 0.8317, 0.831, 0.8323, 0.8332, 0.8478, 0.8707, 0.8332, 0.8303]) ,
                  'GCN':        np.asarray([0.7728, 0.8597, 0.7754, 0.7453, 0.7527, 0.7731, 0.7854, 0.8125, 0.7919, 0.8074, 0.7366, 0.7661, 0.8655, 0.7919, 0.8685, 0.8623, 0.7864, 0.8197, 0.8995, 0.8494, 0.8062, 0.7735, 0.6885, 0.7954, 0.7541, 0.7508, 0.7683, 0.7847, 0.7117, 0.7809]) ,
                  'GraphSAGE':  np.asarray([0.7275, 0.7114, 0.8061, 0.7237, 0.7304, 0.722, 0.6955, 0.7961, 0.7437, 0.7986, 0.7237, 0.81, 0.7815, 0.7046, 0.7315, 0.7398, 0.7524, 0.7657, 0.7906, 0.7679, 0.7597, 0.7115, 0.7015, 0.7159, 0.6921, 0.7043, 0.6872, 0.7308, 0.6955, 0.7101]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9056, 0.9037, 0.9205, 0.9392, 0.9082, 0.9101, 0.9008, 0.915, 0.9098, 0.9244, 0.9108, 0.9263, 0.925, 0.9037, 0.9057, 0.9331, 0.9144, 0.9063, 0.9215, 0.9072, 0.9186, 0.905, 0.9047, 0.9011, 0.904, 0.9056, 0.9276, 0.9176, 0.9059, 0.8963])
                  })

df2 = df2[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df3 = pd.DataFrame({'Metric': ['Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr'],\
                  'GIN':        np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9996, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9996, 1.0, 1.0]) ,
                  'GCN':        np.asarray([0.9967, 0.9877, 0.9926, 0.9965, 0.9966, 0.9975, 0.9951, 0.9937, 0.9943, 0.994, 0.9974, 0.9937, 0.9889, 0.9951, 0.9853, 0.9871, 0.9959, 0.9902, 0.9758, 0.9895, 0.9905, 0.995, 0.9981, 0.9956, 0.997, 0.9974, 0.9966, 0.9906, 0.9986, 0.9946]) ,
                  'GraphSAGE':  np.asarray([0.9973, 0.9982, 0.9885, 0.9973, 0.9982, 0.9987, 0.9977, 0.996, 0.9978, 0.9972, 0.9982, 0.9878, 0.9971, 0.9982, 0.9982, 0.9978, 0.9979, 0.9979, 0.998, 0.9966, 0.9975, 0.9977, 0.9982, 0.9982, 0.9986, 0.9986, 0.9981, 0.9965, 0.9986, 0.9982]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9989, 0.9989, 0.9982, 0.9959, 0.9989, 0.9989, 0.9989, 0.9989, 0.9989, 0.9979, 0.9989, 0.9976, 0.9979, 0.9989, 0.9989, 0.9972, 0.9989, 0.9989, 0.9982, 0.9989, 0.9975, 0.9989, 0.9989, 0.9993, 0.9989, 0.9989, 0.9976, 0.9986, 0.9989, 0.9993])
                  })

df3 = df3[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df4 = pd.DataFrame({'Metric': ['Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc'],\
                  'GIN':        np.asarray([0.8512, 0.8443, 0.8672, 0.8738, 0.8533, 0.8512, 0.828, 0.8645, 0.8521, 0.8928, 0.8491, 0.8651, 0.8599, 0.8346, 0.8599, 0.8687, 0.8642, 0.853, 0.8657, 0.8461, 0.8675, 0.8401, 0.8431, 0.8425, 0.8437, 0.8446, 0.8581, 0.8792, 0.8446, 0.8419]) ,
                  'GCN':        np.asarray([0.7858, 0.8593, 0.7852, 0.7602, 0.7672, 0.7867, 0.7964, 0.8205, 0.8018, 0.816, 0.7527, 0.7774, 0.8657, 0.8024, 0.8654, 0.8611, 0.7979, 0.8244, 0.8855, 0.8512, 0.812, 0.7852, 0.7084, 0.806, 0.7687, 0.766, 0.7816, 0.7925, 0.7304, 0.7919]) ,
                  'GraphSAGE':  np.asarray([0.7443, 0.7298, 0.8105, 0.7407, 0.7476, 0.7401, 0.7148, 0.8069, 0.7596, 0.8102, 0.7413, 0.8136, 0.7943, 0.7235, 0.7485, 0.756, 0.7678, 0.7801, 0.8033, 0.7813, 0.7741, 0.7295, 0.7205, 0.734, 0.712, 0.7235, 0.7072, 0.7467, 0.7154, 0.7286]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9111, 0.9093, 0.9244, 0.9398, 0.9136, 0.9154, 0.9066, 0.9199, 0.9151, 0.9277, 0.916, 0.9292, 0.9283, 0.9093, 0.9111, 0.9352, 0.9193, 0.9117, 0.9253, 0.9127, 0.922, 0.9105, 0.9102, 0.9072, 0.9096, 0.9111, 0.9304, 0.922, 0.9114, 0.9027])
                  })

df4 = df4[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df5 = pd.DataFrame({'Metric': ['F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s'],\
                  'GIN':        np.asarray([0.9132, 0.9088, 0.9233, 0.9274, 0.9146, 0.9132, 0.8983, 0.9216, 0.9138, 0.939, 0.9119, 0.922, 0.9187, 0.9026, 0.9188, 0.9242, 0.9214, 0.9144, 0.9223, 0.91, 0.9235, 0.9062, 0.9081, 0.9077, 0.9085, 0.909, 0.9176, 0.9307, 0.909, 0.9073]) ,
                  'GCN':        np.asarray([0.8706, 0.9193, 0.8707, 0.8528, 0.8577, 0.8711, 0.8779, 0.894, 0.8816, 0.891, 0.8474, 0.8652, 0.9231, 0.8819, 0.9232, 0.9205, 0.8788, 0.8969, 0.9361, 0.9141, 0.8889, 0.8704, 0.8149, 0.8843, 0.8587, 0.8567, 0.8677, 0.8757, 0.8311, 0.8749]) ,
                  'GraphSAGE':  np.asarray([0.8413, 0.8307, 0.8881, 0.8387, 0.8436, 0.8381, 0.8197, 0.8849, 0.8522, 0.8869, 0.839, 0.8901, 0.8762, 0.8261, 0.8443, 0.8497, 0.8579, 0.8665, 0.8822, 0.8675, 0.8625, 0.8306, 0.8239, 0.8338, 0.8176, 0.826, 0.814, 0.8432, 0.82, 0.8298]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.95, 0.9489, 0.9578, 0.9667, 0.9514, 0.9525, 0.9473, 0.9551, 0.9523, 0.9597, 0.9528, 0.9606, 0.9601, 0.9489, 0.95, 0.9641, 0.9548, 0.9503, 0.9583, 0.9509, 0.9564, 0.9497, 0.9495, 0.9477, 0.9491, 0.95, 0.9613, 0.9564, 0.9502, 0.945])
                  })

df5 = df5[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df6 = pd.DataFrame({'Metric': ['ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC'],\
                  'GIN':        np.asarray([0.9202, 0.9165, 0.9288, 0.9323, 0.9213, 0.9202, 0.9077, 0.9273, 0.9207, 0.9404, 0.919, 0.9276, 0.9249, 0.9113, 0.9249, 0.9295, 0.9271, 0.9211, 0.9279, 0.9174, 0.9289, 0.9142, 0.9158, 0.9155, 0.9162, 0.9166, 0.9239, 0.9331, 0.9166, 0.9152]) ,
                  'GCN':        np.asarray([0.8687, 0.8569, 0.8477, 0.855, 0.8587, 0.8733, 0.8661, 0.8709, 0.865, 0.8705, 0.855, 0.8497, 0.8664, 0.8694, 0.8454, 0.8537, 0.8711, 0.8545, 0.7971, 0.8627, 0.8495, 0.8601, 0.8354, 0.8734, 0.8615, 0.8621, 0.8664, 0.8415, 0.8492, 0.8617]) ,
                  'GraphSAGE':  np.asarray([0.8505, 0.8468, 0.8386, 0.8486, 0.8564, 0.8544, 0.8367, 0.8759, 0.8608, 0.8838, 0.853, 0.8361, 0.8753, 0.8434, 0.8569, 0.8588, 0.8652, 0.8718, 0.8842, 0.8663, 0.8665, 0.8446, 0.8418, 0.8491, 0.8394, 0.8455, 0.8347, 0.8477, 0.8411, 0.8462]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9462, 0.9452, 0.9491, 0.9431, 0.9475, 0.9484, 0.9438, 0.9509, 0.9483, 0.9489, 0.9488, 0.9476, 0.9492, 0.9452, 0.9462, 0.9488, 0.9505, 0.9465, 0.9497, 0.947, 0.9437, 0.9458, 0.9457, 0.9461, 0.9454, 0.9462, 0.9483, 0.9499, 0.9463, 0.9437])
                  })

df6 = df6[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df_final = pd.concat([df, df2, df3, df4, df5, df6], ignore_index = True, sort = False)

sns.set(style = "ticks")

dd=pd.melt(df_final,id_vars=['Metric'],value_vars=['GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)'],var_name='Models')
fig, ax = plt.subplots()
ax = sns.boxplot(x='Metric',y='value',data=dd,hue='Models', ax=ax, palette="mako_r", boxprops=dict(alpha=.2), linewidth=0.5, showfliers=False)
#plt.setp(ax.artists,fill=False) 
ylims=ax.get_ylim()
sns.stripplot(x="Metric", y="value", hue='Models', data=dd, dodge=True, palette="mako_r", ax=ax, ec='k', linewidth=0,  size=2.5, alpha = 1)
ax.set(ylim=ylims)
#ax.axhline(y =0.719, ls='--', c='darkred', linewidth = 0.5)
ax.set(xlabel='', ylabel='Score')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#plt.legend([],[],frameon = False)
fig.set_figwidth(6.5)
fig.set_figheight(3.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:4], labels[:4], bbox_to_anchor=(1, 1.02), loc='upper left')
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False, fontsize=10)
plt.savefig('boxplot_metrics_ames_5.png', dpi=2000, bbox_inches='tight')


dd=pd.melt(df_final,id_vars=['Metric'],value_vars=['GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)'],var_name='Models')
fig, ax = plt.subplots()
sns.barplot(x='Metric',y='value',data=dd,hue='Models', ax=ax, palette="YlGnBu", linewidth=0.85, errorbar="sd")
#sns.stripplot(x="Metric", y="value", hue='Models', data=dd, dodge=True, palette="YlGnBu", ax=ax, ec='k', linewidth=0.5,  size=2, alpha = 0.7)
sns.set(style = 'white')
#ax.axhline(y =0.719, ls='--', c='darkred', linewidth = 0.5)
ax.set(xlabel='', ylabel='Score')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#plt.legend([],[],frameon = False)
fig.set_figwidth(10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:4], labels[:4], bbox_to_anchor=(1, 1.02), loc='upper left')
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False, fontsize=10)
plt.savefig('barplot_metrics_ames_5.png', dpi=300, bbox_inches='tight')

###########################################################################################

df = pd.DataFrame({'Metric': ['Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp','Sp'],\
                  'GIN':        np.asarray([1.0, 0.9819, 0.9819, 1.0, 0.9729, 1.0, 0.991, 0.9955, 0.9864, 0.9819, 0.991, 1.0, 1.0, 0.9819, 0.9955, 1.0, 0.991, 1.0, 0.9818, 1.0, 1.0, 0.991, 0.9864, 0.9955, 1.0, 0.991, 1.0, 0.9955, 0.9955, 0.9819]) ,
                  'GCN':        np.asarray([0.9409, 0.9729, 0.914, 0.9593, 0.8281, 0.8914, 0.7014, 0.9231, 0.9091, 0.8416, 0.9095, 0.9005, 0.9364, 0.9502, 0.819, 0.8959, 0.9186, 0.8869, 0.7909, 0.9591, 0.8552, 0.819, 0.9729, 0.8416, 0.8818, 0.9276, 0.9231, 0.8959, 0.9774, 0.8507]) ,
                  'GraphSAGE':  np.asarray([0.9773, 0.9819, 0.8597, 0.9819, 0.9819, 0.9729, 0.7692, 0.9638, 0.9727, 0.9593, 0.9412, 0.9819, 0.9773, 0.9502, 0.9729, 0.9502, 0.9638, 0.9548, 0.9227, 0.9773, 0.8869, 0.8597, 0.9502, 0.914, 0.9545, 0.9819, 0.9186, 0.9864, 0.9864, 0.9729]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9909, 0.9864, 0.9819, 0.991, 0.9276, 0.9955, 0.9864, 0.991, 0.9909, 0.991, 0.9864, 0.991, 0.9909, 0.991, 0.991, 0.9683, 0.991, 0.991, 0.9909, 0.9864, 0.991, 0.991, 0.9683, 0.9864, 0.9864, 0.9955, 0.9638, 0.991, 0.991, 0.9864])
                  })


df = df[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df2 = pd.DataFrame({'Metric': ['Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn','Sn'],\
                  'GIN':        np.asarray([0.8294, 0.8297, 0.8478, 0.8488, 0.8838, 0.8663, 0.8475, 0.8702, 0.8735, 0.8537, 0.8679, 0.8265, 0.8304, 0.8372, 0.8375, 0.864, 0.8388, 0.8404, 0.8288, 0.8385, 0.8546, 0.8812, 0.8556, 0.8262, 0.8599, 0.8184, 0.8725, 0.8676, 0.8601, 0.8835]) ,
                  'GCN':        np.asarray([0.7861, 0.7378, 0.8084, 0.7478, 0.8689, 0.8653, 0.887, 0.8304, 0.8249, 0.8767, 0.8398, 0.831, 0.8214, 0.7922, 0.8789, 0.8203, 0.8135, 0.7886, 0.8421, 0.7693, 0.8598, 0.8818, 0.7028, 0.8521, 0.8243, 0.7802, 0.8142, 0.7996, 0.7915, 0.8534]) ,
                  'GraphSAGE':  np.asarray([0.7301, 0.7235, 0.8028, 0.7468, 0.7656, 0.7345, 0.8592, 0.7834, 0.7673, 0.7886, 0.7423, 0.7248, 0.7408, 0.7672, 0.7714, 0.776, 0.7604, 0.7608, 0.8188, 0.7272, 0.8139, 0.8469, 0.7391, 0.8019, 0.7725, 0.7174, 0.8074, 0.685, 0.7345, 0.7679]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.8913, 0.909, 0.9077, 0.8958, 0.9365, 0.9032, 0.9074, 0.898, 0.9052, 0.9016, 0.9048, 0.8899, 0.8903, 0.9029, 0.8925, 0.9204, 0.8964, 0.8899, 0.9055, 0.9036, 0.8964, 0.91, 0.9204, 0.9061, 0.9065, 0.8919, 0.9259, 0.9006, 0.8993, 0.9058])
                  })

df2 = df2[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df3 = pd.DataFrame({'Metric': ['Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr','Pr'],\
                  'GIN':        np.asarray([1.0, 0.9984, 0.9985, 1.0, 0.9978, 1.0, 0.9992, 0.9996, 0.9989, 0.9985, 0.9993, 1.0, 1.0, 0.9985, 0.9996, 1.0, 0.9992, 1.0, 0.9984, 1.0, 1.0, 0.9993, 0.9989, 0.9996, 1.0, 0.9992, 1.0, 0.9996, 0.9996, 0.9985]) ,
                  'GCN':        np.asarray([0.9947, 0.9974, 0.9924, 0.9961, 0.986, 0.9911, 0.9765, 0.9934, 0.9922, 0.9872, 0.9923, 0.9915, 0.9945, 0.9955, 0.9855, 0.991, 0.9929, 0.9898, 0.9826, 0.9962, 0.9881, 0.9855, 0.9972, 0.9869, 0.9899, 0.9934, 0.9933, 0.9908, 0.998, 0.9876]) ,
                  'GraphSAGE':  np.asarray([0.9978, 0.9982, 0.9877, 0.9983, 0.9983, 0.9974, 0.9811, 0.9967, 0.9975, 0.9963, 0.9944, 0.9982, 0.9978, 0.9954, 0.9975, 0.9954, 0.9966, 0.9958, 0.9933, 0.9978, 0.9902, 0.9883, 0.9952, 0.9924, 0.9958, 0.9982, 0.9928, 0.9986, 0.9987, 0.9975]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9993, 0.9989, 0.9986, 0.9993, 0.9945, 0.9996, 0.9989, 0.9993, 0.9993, 0.9993, 0.9989, 0.9993, 0.9993, 0.9993, 0.9993, 0.9975, 0.9993, 0.9993, 0.9993, 0.9989, 0.9993, 0.9993, 0.9975, 0.9989, 0.9989, 0.9996, 0.9972, 0.9993, 0.9993, 0.9989])
                  })

df3 = df3[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df4 = pd.DataFrame({'Metric': ['Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc','Acc'],\
                  'GIN':        np.asarray([0.8408, 0.8399, 0.8568, 0.8589, 0.8897, 0.8752, 0.8571, 0.8785, 0.881, 0.8622, 0.8761, 0.8381, 0.8417, 0.8468, 0.848, 0.8731, 0.8489, 0.8511, 0.839, 0.8492, 0.8644, 0.8885, 0.8644, 0.8375, 0.8692, 0.8299, 0.881, 0.8761, 0.8692, 0.89]) ,
                  'GCN':        np.asarray([0.7964, 0.7535, 0.8154, 0.7619, 0.8662, 0.8671, 0.8746, 0.8366, 0.8305, 0.8743, 0.8444, 0.8356, 0.829, 0.8027, 0.8749, 0.8254, 0.8205, 0.7952, 0.8387, 0.7819, 0.8595, 0.8776, 0.7208, 0.8514, 0.8281, 0.79, 0.8215, 0.806, 0.8039, 0.8532]) ,
                  'GraphSAGE':  np.asarray([0.7465, 0.7408, 0.8066, 0.7625, 0.7801, 0.7505, 0.8532, 0.7955, 0.781, 0.8, 0.7556, 0.742, 0.7565, 0.7795, 0.7849, 0.7876, 0.774, 0.7737, 0.8257, 0.7438, 0.8187, 0.8477, 0.7532, 0.8094, 0.7846, 0.735, 0.8148, 0.7051, 0.7514, 0.7816]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.8979, 0.9142, 0.9127, 0.9021, 0.936, 0.9094, 0.9127, 0.9042, 0.9109, 0.9076, 0.9103, 0.8967, 0.897, 0.9088, 0.8991, 0.9236, 0.9027, 0.8967, 0.9112, 0.9091, 0.9027, 0.9154, 0.9236, 0.9115, 0.9118, 0.8988, 0.9284, 0.9066, 0.9054, 0.9112])
                  })

df4 = df4[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df5 = pd.DataFrame({'Metric': ['F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s','F1s'],\
                  'GIN':        np.asarray([0.9068, 0.9063, 0.917, 0.9182, 0.9373, 0.9284, 0.9171, 0.9304, 0.932, 0.9204, 0.929, 0.905, 0.9074, 0.9107, 0.9114, 0.9271, 0.912, 0.9133, 0.9057, 0.9122, 0.9216, 0.9365, 0.9217, 0.9046, 0.9247, 0.8998, 0.9319, 0.9289, 0.9247, 0.9375]) ,
                  'GCN':        np.asarray([0.8782, 0.8482, 0.891, 0.8543, 0.9238, 0.924, 0.9296, 0.9046, 0.9009, 0.9287, 0.9097, 0.9042, 0.8997, 0.8823, 0.9292, 0.8976, 0.8943, 0.8778, 0.9069, 0.8682, 0.9195, 0.9308, 0.8245, 0.9145, 0.8995, 0.874, 0.8949, 0.885, 0.8828, 0.9156]) ,
                  'GraphSAGE':  np.asarray([0.8432, 0.839, 0.8857, 0.8544, 0.8666, 0.846, 0.9161, 0.8773, 0.8674, 0.8804, 0.85, 0.8398, 0.8503, 0.8665, 0.87, 0.8721, 0.8627, 0.8625, 0.8976, 0.8413, 0.8934, 0.9121, 0.8482, 0.887, 0.8701, 0.8348, 0.8906, 0.8126, 0.8465, 0.8678]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9422, 0.9519, 0.951, 0.9447, 0.9647, 0.949, 0.951, 0.946, 0.9499, 0.9479, 0.9495, 0.9414, 0.9416, 0.9486, 0.9429, 0.9574, 0.9451, 0.9414, 0.9501, 0.9489, 0.9451, 0.9526, 0.9574, 0.9503, 0.9505, 0.9427, 0.9602, 0.9474, 0.9467, 0.9501])
                  })

df5 = df5[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df6 = pd.DataFrame({'Metric': ['ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC','ROC-AUC'],\
                  'GIN':        np.asarray([0.9147, 0.9058, 0.9149, 0.9244, 0.9283, 0.9331, 0.9192, 0.9328, 0.9299, 0.9178, 0.9294, 0.9132, 0.9152, 0.9095, 0.9165, 0.932, 0.9149, 0.9202, 0.9053, 0.9193, 0.9273, 0.9361, 0.921, 0.9108, 0.9299, 0.9047, 0.9362, 0.9315, 0.9278, 0.9327]) ,
                  'GCN':        np.asarray([0.8635, 0.8553, 0.8612, 0.8535, 0.8485, 0.8784, 0.7942, 0.8767, 0.867, 0.8591, 0.8746, 0.8657, 0.8789, 0.8712, 0.849, 0.8581, 0.866, 0.8377, 0.8165, 0.8642, 0.8575, 0.8504, 0.8378, 0.8468, 0.853, 0.8539, 0.8686, 0.8478, 0.8844, 0.852]) ,
                  'GraphSAGE':  np.asarray([0.8537, 0.8527, 0.8313, 0.8644, 0.8738, 0.8537, 0.8142, 0.8736, 0.87, 0.8739, 0.8417, 0.8534, 0.859, 0.8587, 0.8721, 0.8631, 0.8621, 0.8578, 0.8707, 0.8522, 0.8504, 0.8533, 0.8447, 0.858, 0.8635, 0.8496, 0.863, 0.8357, 0.8605, 0.8704]) ,
                  'Meta-GTMP (GIN+TR)': np.asarray([0.9411, 0.9477, 0.9448, 0.9434, 0.9321, 0.9493, 0.9469, 0.9445, 0.948, 0.9463, 0.9456, 0.9404, 0.9406, 0.9469, 0.9417, 0.9443, 0.9437, 0.9404, 0.9482, 0.945, 0.9437, 0.9505, 0.9443, 0.9463, 0.9464, 0.9437, 0.9448, 0.9458, 0.9451, 0.9461])
                  })

df6 = df6[['Metric','GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)']]

df_final = pd.concat([df, df2, df3, df4, df5, df6], ignore_index = True, sort = False)

sns.set(style = "ticks")

dd=pd.melt(df_final,id_vars=['Metric'],value_vars=['GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)'],var_name='Models')
fig, ax = plt.subplots()
ax = sns.boxplot(x='Metric',y='value',data=dd,hue='Models', ax=ax, palette="mako_r", boxprops=dict(alpha=.2), linewidth=0.5, showfliers=False)
#plt.setp(ax.artists,fill=False) 
ylims=ax.get_ylim()
sns.stripplot(x="Metric", y="value", hue='Models', data=dd, dodge=True, palette="mako_r", ax=ax, ec='k', linewidth=0,  size=2.5, alpha = 1)
ax.set(ylim=ylims)
#ax.axhline(y =0.719, ls='--', c='darkred', linewidth = 0.5)
ax.set(xlabel='', ylabel='Score')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#plt.legend([],[],frameon = False)
fig.set_figwidth(6.5)
fig.set_figheight(3.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:4], labels[:4], bbox_to_anchor=(1, 1.02), loc='upper left')
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False, fontsize=10)
plt.savefig('boxplot_metrics_ames_10.png', dpi=2000, bbox_inches='tight')

dd=pd.melt(df_final,id_vars=['Metric'],value_vars=['GIN','GCN','GraphSAGE','Meta-GTMP (GIN+TR)'],var_name='Models')
fig, ax = plt.subplots()
ec = ['red', 'gray', 'black', 'blue']
sns.barplot(x='Metric',y='value',data=dd,hue='Models', ax=ax, palette="YlGnBu", linewidth=0.85, errorbar="sd", capsize = 0.05, errwidth = 3, ci = 'sd')
#sns.stripplot(x="Metric", y="value", hue='Models', data=dd, dodge=True, palette="YlGnBu", ax=ax, ec='k', linewidth=0.5,  size=2, alpha = 0.7)
sns.set(style = 'whitegrid')
#ax.axhline(y =0.719, ls='--', c='darkred', linewidth = 0.5)
ax.set(xlabel='', ylabel='Score')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#plt.legend([],[],frameon = False)
fig.set_figwidth(10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:4], labels[:4], bbox_to_anchor=(1, 1.02), loc='upper left')
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False, fontsize=10)
plt.savefig('barplot_metrics_ames_10.png', dpi=300, bbox_inches='tight')


##### SINGLE-TASK MODELS #####


df = pd.DataFrame({'Experiment': ['5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot','5-shot'],\
                  'ST1-TA98':        np.asarray([0.872, 0.8643, 0.8658, 0.869, 0.8633, 0.8711, 0.8672, 0.8686, 0.8633, 0.8676, 0.87, 0.869, 0.867, 0.8602, 0.8622, 0.854, 0.8631, 0.8675, 0.8701, 0.8644, 0.8639, 0.8649, 0.8615, 0.8606, 0.8667, 0.8644, 0.8633, 0.8711, 0.8634, 0.8668]),
                  'ST2-TA100':        np.asarray([0.8624, 0.868, 0.8496, 0.8773, 0.8809, 0.8643, 0.8811, 0.8648, 0.8485, 0.8315, 0.8865, 0.863, 0.824, 0.8801, 0.8581, 0.8117, 0.8809, 0.8636, 0.8429, 0.8545, 0.8276, 0.8814, 0.8655, 0.8866, 0.8792, 0.8825, 0.8647, 0.8618, 0.8786, 0.8875]),
                  'ST3-TA102':        np.asarray([0.7999, 0.8299, 0.8188, 0.8024, 0.8555, 0.8422, 0.7429, 0.8153, 0.8257, 0.8271, 0.815, 0.8498, 0.8222, 0.7864, 0.7966, 0.8332, 0.8236, 0.8299, 0.8437, 0.8192, 0.8049, 0.7945, 0.8221, 0.8328, 0.755, 0.8156, 0.7842, 0.8263, 0.7922, 0.8179]),
                  'ST4-TA1535':        np.asarray([0.7338, 0.7457, 0.773, 0.7948, 0.7429, 0.7824, 0.7216, 0.753, 0.7976, 0.7793, 0.7399, 0.7323, 0.805, 0.801, 0.7698, 0.8044, 0.8078, 0.7323, 0.8049, 0.783, 0.7391, 0.7838, 0.7432, 0.7793, 0.7734, 0.8012, 0.7272, 0.7639, 0.8022, 0.7772]),
                  'ST5-TA1537':        np.asarray([0.7747, 0.7622, 0.7758, 0.7846, 0.7479, 0.793, 0.5774, 0.767, 0.7442, 0.722, 0.7808, 0.7784, 0.7532, 0.7688, 0.7895, 0.7473, 0.7751, 0.7788, 0.7945, 0.4987, 0.787, 0.7577, 0.8015, 0.7746, 0.7984, 0.7622, 0.7909, 0.7792, 0.7346, 0.7837]),
                  'All-strains': np.asarray([0.9462, 0.9452, 0.9491, 0.9431, 0.9475, 0.9484, 0.9438, 0.9509, 0.9483, 0.9489, 0.9488, 0.9476, 0.9492, 0.9452, 0.9462, 0.9488, 0.9505, 0.9465, 0.9497, 0.947, 0.9437, 0.9458, 0.9457, 0.9461, 0.9454, 0.9462, 0.9483, 0.9499, 0.9463, 0.9437])
                          
                  })


df = df[['Experiment','ST1-TA98','ST2-TA100','ST3-TA102','ST4-TA1535','ST5-TA1537', 'All-strains']]

df2 = pd.DataFrame({'Experiment': ['10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot','10-shot'],\
                  'ST1-TA98':        np.asarray([0.8611, 0.87, 0.8727, 0.8641, 0.7987, 0.8293, 0.8691, 0.8, 0.8595, 0.8674, 0.8441, 0.7963, 0.8692, 0.8633, 0.863, 0.873, 0.8645, 0.8688, 0.8704, 0.8634, 0.8683, 0.8139, 0.8735, 0.872, 0.8622, 0.8693, 0.7965, 0.8398, 0.8533, 0.864]),
                  'ST2-TA100':        np.asarray([0.8794, 0.8787, 0.8756, 0.8788, 0.8721, 0.8734, 0.8769, 0.8697, 0.873, 0.8798, 0.8745, 0.8776, 0.8736, 0.8727, 0.8687, 0.8743, 0.8761, 0.8703, 0.8687, 0.8721, 0.8708, 0.8658, 0.8673, 0.8793, 0.8727, 0.879, 0.8703, 0.869, 0.875, 0.8789]),
                  'ST3-TA102':        np.asarray([0.8496, 0.8377, 0.871, 0.8354, 0.8397, 0.82, 0.8417, 0.8448, 0.8584, 0.8515, 0.8642, 0.8448, 0.8636, 0.8543, 0.8475, 0.8566, 0.856, 0.8731, 0.8527, 0.8588, 0.8475, 0.8372, 0.8359, 0.854, 0.8395, 0.8513, 0.8557, 0.8495, 0.8561, 0.852]),
                  'ST4-TA1535':        np.asarray([0.7251, 0.7676, 0.731, 0.741, 0.7812, 0.7457, 0.7973, 0.743, 0.7708, 0.7713, 0.7574, 0.7795, 0.7761, 0.7522, 0.7691, 0.7984, 0.7886, 0.7855, 0.7848, 0.781, 0.7919, 0.7816, 0.7684, 0.7611, 0.7601, 0.7481, 0.7808, 0.7658, 0.7524, 0.7448]),
                  'ST5-TA1537':        np.asarray([0.7855, 0.7711, 0.7688, 0.7926, 0.7632, 0.7816, 0.7966, 0.7861, 0.7806, 0.7163, 0.7545, 0.7729, 0.7844, 0.7981, 0.7848, 0.7766, 0.7978, 0.7907, 0.7833, 0.791, 0.7955, 0.6881, 0.8006, 0.7916, 0.7849, 0.7404, 0.7083, 0.7958, 0.7847, 0.6414]),
                  'All-strains': np.asarray([0.9411, 0.9477, 0.9448, 0.9434, 0.9321, 0.9493, 0.9469, 0.9445, 0.948, 0.9463, 0.9456, 0.9404, 0.9406, 0.9469, 0.9417, 0.9443, 0.9437, 0.9404, 0.9482, 0.945, 0.9437, 0.9505, 0.9443, 0.9463, 0.9464, 0.9437, 0.9448, 0.9458, 0.9451, 0.9461])
                  
                  })


df2 = df2[['Experiment','ST1-TA98','ST2-TA100','ST3-TA102','ST4-TA1535','ST5-TA1537','All-strains']]

df_final = pd.concat([df, df2], ignore_index = True, sort = False)

sns.set(style = "ticks")

dd=pd.melt(df_final,id_vars=['Experiment'],value_vars=['ST1-TA98','ST2-TA100','ST3-TA102','ST4-TA1535','ST5-TA1537','All-strains'],var_name='Models')
fig, ax = plt.subplots()
#colors = ["#ff8100","#fb9b50","#ffb347","#9fc0de","#0466c8", "#023e7d"]
ax = sns.barplot(x='value',y='Experiment',data=dd,hue='Models', ax=ax, palette="ch:start=.1,rot=-.2", saturation = 0.85, linewidth=0.85, errorbar="sd", capsize = 0.05, errwidth = 2)
#sns.stripplot(x="Metric", y="value", hue='Models', data=dd, dodge=True, palette="YlGnBu", ax=ax, ec='k', linewidth=0.5,  size=2, alpha = 0.7)
sns.set(style = 'whitegrid')
#ax.axhline(y =0.719, ls='--', c='darkred', linewidth = 0.5)
ax.set(xlabel='ROC-AUC score', ylabel='')
for container in ax.containers:
    ax.bar_label(container, fmt= '%.4f', padding=10)
xlims=ax.get_xlim()
ax.set(xlim=(0, 1.1))
ax.spines[['right', 'top']].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#plt.legend([],[],frameon = False)
fig.set_figwidth(5)
fig.set_figheight(5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:6], labels[:6], bbox_to_anchor=(1, 1.02), loc='upper left')
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=6, title=None, frameon=False, fontsize=10)
plt.savefig('barplot_ST_models.png', dpi=600, bbox_inches='tight')

sns.set(style = "ticks")

dd=pd.melt(df_final,id_vars=['Experiment'],value_vars=['ST1-TA98','ST2-TA100','ST3-TA102','ST4-TA1535','ST5-TA1537','All-strains'],var_name='Models')
fig, ax = plt.subplots()
ax = sns.boxplot(x='Experiment',y='value',data=dd,hue='Models', ax=ax, palette="rocket_r", boxprops=dict(alpha=.1), linewidth=0.5, showfliers=False)
#plt.setp(ax.artists,fill=False) 
ylims=ax.get_ylim()
sns.stripplot(x="Experiment", y="value", hue='Models', data=dd, dodge=True, palette="rocket_r", ax=ax, ec='k', linewidth=0,  size=3, alpha = 1)
ax.set(ylim=ylims)
#ax.axhline(y =0.719, ls='--', c='darkred', linewidth = 0.5)
ax.set(xlabel='', ylabel='ROC-AUC')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
#plt.legend([],[],frameon = False)
fig.set_figwidth(8)
fig.set_figheight(3.5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:6], labels[:6], bbox_to_anchor=(1, 1.05), loc='upper left')
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.05), ncol=6, title=None, frameon=False, fontsize=10)
plt.savefig('boxplot_ST_models.png', dpi=2000, bbox_inches='tight')

