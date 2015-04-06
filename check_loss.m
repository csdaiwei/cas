function [loss, accuracy] = check_loss( samples, labels, w)
%%calculate loss(hinge) and accuracy
p = samples * w;
loss = sum(max(0, 1-p));
accuracy = mean(sign(p) == labels);
end
