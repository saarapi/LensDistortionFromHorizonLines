%Este script carga el ground truth del horizonte, definido como una recta por frame, y 
% genera una imagen binaria acumulada dibujando únicamente una de cada N rectas para simular 
% un submuestreo temporal del horizonte.

clear all
clc

matFile = 'MVI_0804_VIS_OB_HorizonGT.mat';
N = 30;

outImage = sprintf('horizontes_GT_submuestreo_%d.png', N);

W = 1920;
H = 1080;

S = load(matFile);

if ~isfield(S,'structXML')
    error('No existe la variable structXML en el .mat')
end

G = S.structXML;

if ~isstruct(G) || ~all(isfield(G,{'X','Y','Nx','Ny'}))
    error('structXML no tiene campos X,Y,Nx,Ny como se espera')
end

X = double([G.X]).';
Y = double([G.Y]).';
Nx = double([G.Nx]).';
Ny = double([G.Ny]).';

nTotal = numel(X);
BW = false(H, W);
nDrawn = 0;

for i = 1:N:nTotal
    x0 = X(i);
    y0 = Y(i);
    a = Nx(i);
    b = Ny(i);
    c = -(a*x0 + b*y0);

    pts = zeros(0,2);

    if abs(b) > 1e-12
        y = -(a*1 + c)/b;
        if y >= 1 && y <= H
            pts(end+1,:) = [1 y];
        end
        y = -(a*W + c)/b;
        if y >= 1 && y <= H
            pts(end+1,:) = [W y];
        end
    end

    if abs(a) > 1e-12
        x = -(b*1 + c)/a;
        if x >= 1 && x <= W
            pts(end+1,:) = [x 1];
        end
        x = -(b*H + c)/a;
        if x >= 1 && x <= W
            pts(end+1,:) = [x H];
        end
    end

    if size(pts,1) < 2
        continue
    end

    keep = true(size(pts,1),1);
    for p = 1:size(pts,1)
        if ~keep(p), continue, end
        for q = p+1:size(pts,1)
            if keep(q)
                if sum((pts(p,:) - pts(q,:)).^2) < 1e-8
                    keep(q) = false;
                end
            end
        end
    end
    pts = pts(keep,:);

    if size(pts,1) < 2
        continue
    end

    dmax = -inf;
    p1 = pts(1,:);
    p2 = pts(2,:);
    for m = 1:size(pts,1)
        for n = m+1:size(pts,1)
            d = sum((pts(m,:) - pts(n,:)).^2);
            if d > dmax
                dmax = d;
                p1 = pts(m,:);
                p2 = pts(n,:);
            end
        end
    end

    x1 = p1(1); y1 = p1(2);
    x2 = p2(1); y2 = p2(2);

    npts = max(abs(round(x2)-round(x1)), abs(round(y2)-round(y1))) + 1;
    xs = round(linspace(x1, x2, npts));
    ys = round(linspace(y1, y2, npts));

    valid = xs>=1 & xs<=W & ys>=1 & ys<=H;
    xs = xs(valid);
    ys = ys(valid);

    if isempty(xs)
        continue
    end

    idx = sub2ind([H W], ys, xs);
    BW(idx) = true;

    nDrawn = nDrawn + 1;
end

imwrite(uint8(BW)*255, outImage);

fprintf('Rectas totales en el GT: %d\n', nTotal);
fprintf('Submuestreo N = %d\n', N);
fprintf('Rectas dibujadas: %d\n', nDrawn);
fprintf('Salida: %s\n', outImage);

figure
imshow(BW)
title(sprintf('Horizontes GT submuestreo N=%d | dibujadas=%d', N, nDrawn))
