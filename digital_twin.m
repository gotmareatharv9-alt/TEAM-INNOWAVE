clc; 
clear; 
close all; 
% time and link parameters 
time = 0:0.05:20; 
link1 = 1.0; 
link2 = 0.85; 
link3 = 0.65; 
% joint angle generation 
theta1 = deg2rad(20*sin(0.3*time)); 
theta2 = deg2rad(15*sin(0.3*time + 1)); 
theta3 = deg2rad(10*sin(0.3*time + 2)); 
jointAngle = [theta1' theta2' theta3']; 
% healthy time limit 
faultStartTime = 12; 
healthyIndex = time <= faultStartTime; 
% reduced order model using healthy data 
healthyMatrix = jointAngle(healthyIndex,:)'; 
[Umat,Smat,~] = svd(healthyMatrix,'econ'); 
energyVal = cumsum(diag(Smat).^2) / sum(diag(Smat).^2); 
orderROM = find(energyVal >= 0.99,1); 
reducedBase = Umat(:,1:orderROM); 
% reduced coordinates 
reducedState = reducedBase' * jointAngle'; 
% neural network digital twin 
dtNet = fitnet(8); 
dtNet = train(dtNet, reducedState(:,healthyIndex), reducedState(:,healthyIndex)); 
% adjust third joint response after fault initiation 
idx = find(time > faultStartTime); 
for k = idx 
jointAngle(k, 3) = 1.8 * jointAngle(k, 3); 
end 
% digital twin prediction 
faultReduced = reducedBase' * jointAngle'; 
predictedReduced = dtNet(faultReduced); 
% reconstruct joint angles 
twinJoint = (reducedBase * predictedReduced)'; 
% fault detection 
difference = abs(jointAngle - twinJoint); 
limitAngle = deg2rad(5); 
faultDetected = difference > limitAngle; 
% plot for joint 3 
figure; 
plot(time, rad2deg(jointAngle(:,3)), 'LineWidth', 1.6); 
hold on; 
plot(time, rad2deg(twinJoint(:,3)), '--' , 'LineWidth', 1.6); 
xlabel('Time in seconds'); 
ylabel('Angle in degrees'); 
legend('Actual Joint','Digital Twin'); 
title('Joint 3 Fault Detection'); 
grid on; 
% 3D robotic arm animation 
figure; 
for k = 1:length(time) 
baseX = 0; baseY = 0; baseZ = 0; 
x1 = link1*cos(jointAngle(k,1))*cos(jointAngle(k,2)); 
y1 = link1*sin(jointAngle(k,1))*cos(jointAngle(k,2)); 
z1 = link1*sin(jointAngle(k,2)); 
x2 = x1 + link2*cos(jointAngle(k,1))*cos(jointAngle(k,2)+jointAngle(k,3)); 
y2 = y1 + link2*sin(jointAngle(k,1))*cos(jointAngle(k,2)+jointAngle(k,3)); 
z2 = z1 + link2*sin(jointAngle(k,2)+jointAngle(k,3)); 
x3 = x2 + link3*cos(jointAngle(k,1))*cos(jointAngle(k,2)+jointAngle(k,3)); 
y3 = y2 + link3*sin(jointAngle(k,1))*cos(jointAngle(k,2)+jointAngle(k,3)); 
z3 = z2 + link3*sin(jointAngle(k,2)+jointAngle(k,3)); 
plot3([baseX x1 x2 x3] ,[baseY y1 y2 y3] ,[baseZ z1 z2 z3] , 'o-', 'LineWidth', 2.2); 
axis equal; 
set(gca,'XGrid','on','YGrid','on','ZGrid','on'); 
xlim([-2.2 2.2]); 
ylim([-2.2 2.2]); 
zlim([0 2.2]); 
xlabel("X Axis"); 
ylabel("Y Axis"); 
zlabel("Z Axis"); 
title('3 Degree-Of-Freedom Robotic Arm Animation'); 
drawnow; 
end 
% display result 
disp(['Reduced order selected: ', num2str(orderROM)]); 
if any(faultDetected(:)) 
disp('Fault detected in joint 3 after 12 seconds'); 
else 
disp('No fault detected'); 
end 