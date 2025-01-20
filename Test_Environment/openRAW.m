
filename = "C:\CODE\SensiaNuc_STL\Test_Environment\Calibration\01\Nucs\2750\GainU.raw"; % Cambia por el nombre de tu archivo
rows = 512; 
cols = 640; 


fid = fopen(filename, 'rb');
if fid == -1
    error('No se pudo abrir el archivo.');
end
data = fread(fid, rows * cols, 'float32'); % Leemos
fclose(fid);


matrix = reshape(data, [cols, rows])'; % Transponer si las filas y columnas están invertidas


disp(matrix); 
figure;
imagesc(matrix); 
caxis([0.7, 1.3]);
colorbar; 
axis image; 

%% Check: se localiza un valor en la matriz
N = 1.00804;% Valor sacado de C++
[~,idx] = min(abs(matrix(:)-N));
[row, col] = ind2sub(size(matrix), idx);
fprintf('El valor más cercano a %.5f está en (i,j) = (%d, %d) y es %.5f\n', ...
    N, row, col, matrix(row, col));

%%

filename = "C:\RAWs\Gas\gasMethabeTotal_optic35mm_dist25m_g_s.raw"; % Cambia por el nombre de tu archivo
rows = 512; 
cols = 640; 

fid = fopen(filename, 'rb');
if fid == -1
    error('No se pudo abrir el archivo.');
end

data = []; % Inicializar una matriz vacía para almacenar los datos
layer_size = rows * cols; % Tamaño de una capa
i = 0;
while ~feof(fid)
    i = i + 1;
    disp(i);
    % Leer una capa de datos
    layer_data = fread(fid, layer_size, 'float32');
    if isempty(layer_data) % Salir si no se pudo leer más datos
        break;
    end
    % Verificar si la capa tiene el tamaño esperado
    if numel(layer_data) ~= layer_size
        warning('La última capa tiene un tamaño incompleto y será descartada.');
        break;
    end
    % Reshape y almacenar la capa
    layer = reshape(layer_data, [cols, rows])'; % Transponer si es necesario
    data = cat(3, data, layer); % Concatenar la capa en la tercera dimensión
    if i > 10
        break;
    end
end

fclose(fid);

disp(size(data)); % Mostrar el tamaño de la matriz 3D

% Visualizar una capa como ejemplo
figure;
imagesc(data(:, :, 1)); % Mostrar la primera capa
%%caxis([0.7, 1.3]);
colorbar;
axis image;
