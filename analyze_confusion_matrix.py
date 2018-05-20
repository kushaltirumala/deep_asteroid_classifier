from __future__ import division

tn, fp, fn, tp = 416, 24, 31, 165

total = tn + fp + fn + tp

accuracy = (tp + tn)/(total)
misclassification_rate = (fp + fn)/(total)
tp_rate= (tp)/(tp + fn)
fp_rate = (fp)/(tn + fp)
specificty = (tn)/(tn + fp)
precision = (tp)/(tp + fp)
prevalence = (fn + tp)/(total)

print("accuracy: %f" % accuracy)
print("misclassification_rate:%f" % misclassification_rate)
print("tp_rate: %f" % tp_rate)
print("fp_rate: %f " % fp_rate)
print("specificty: %f" % specificty)
print("precision: %f" % precision)
print("prevalence: %f"% prevalence)
