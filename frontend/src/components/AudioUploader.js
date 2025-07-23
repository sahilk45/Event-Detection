import React, { useState, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  LinearProgress,
  Fade,
} from '@mui/material';
import {
  CloudUpload,
  GraphicEq,
  CheckCircle,
} from '@mui/icons-material';

const AudioUploader = ({ onFileSelect, isProcessing }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file) => {
    if (file && file.type.startsWith('audio/')) {
      setSelectedFile(file);
      onFileSelect(file);
    } else {
      alert('Please select an audio file');
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const formatFileSize = (bytes) => {
    return (bytes / (1024 * 1024)).toFixed(2);
  };

  return (
    <Card
      sx={{
        position: 'relative',
        background: (theme) => 
          dragActive 
            ? `linear-gradient(135deg, ${theme.palette.primary.main}15, ${theme.palette.secondary.main}15)`
            : 'transparent',
        border: (theme) => 
          dragActive 
            ? `2px dashed ${theme.palette.primary.main}` 
            : `2px dashed ${theme.palette.grey[500]}`,
        transition: 'all 0.3s ease-in-out',
        '&:hover': {
          borderColor: 'primary.main',
          transform: 'scale(1.01)',
        },
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleInputChange}
        style={{ display: 'none' }}
        disabled={isProcessing}
      />
      
      <CardContent
        onClick={!isProcessing ? triggerFileInput : undefined}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        sx={{
          minHeight: 200,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          cursor: isProcessing ? 'not-allowed' : 'pointer',
          opacity: isProcessing ? 0.7 : 1,
          p: 4,
        }}
      >
        {selectedFile ? (
          <Box sx={{ width: '100%' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <CheckCircle color="success" sx={{ fontSize: 40 }} />
              <Box sx={{ flex: 1, textAlign: 'left' }}>
                <Typography variant="h6" color="success.main" gutterBottom>
                  {selectedFile.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {formatFileSize(selectedFile.size)} MB
                </Typography>
              </Box>
            </Box>
            
            {isProcessing && (
              <Box sx={{ width: '100%', mt: 2 }}>
                <LinearProgress 
                  color="primary" 
                  sx={{ borderRadius: 2, height: 6 }}
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Analyzing audio with AI models...
                </Typography>
              </Box>
            )}
          </Box>
        ) : (
          <Box>
            <CloudUpload 
              sx={{ 
                fontSize: 64, 
                color: 'primary.main',
                mb: 2,
              }} 
            />
            
            <Typography variant="h5" gutterBottom fontWeight={600}>
              Drop your audio file here
            </Typography>
            
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              or click to browse files
            </Typography>
            
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
              {['WAV', 'MP3', 'M4A', 'FLAC', 'AAC', 'OGG'].map((format) => (
                <Chip 
                  key={format}
                  label={format}
                  size="small"
                  variant="outlined"
                  color="primary"
                />
              ))}
            </Box>
          </Box>
        )}
      </CardContent>
      
      {selectedFile && !isProcessing && (
        <Fade in={!isProcessing}>
          <Box sx={{ p: 2, pt: 0 }}>
            <Button
              variant="contained"
              fullWidth
              size="large"
              startIcon={<GraphicEq />}
              onClick={() => onFileSelect(selectedFile)}
              sx={{ py: 1.5 }}
            >
              Analyze Audio
            </Button>
          </Box>
        </Fade>
      )}
    </Card>
  );
};

export default AudioUploader;
