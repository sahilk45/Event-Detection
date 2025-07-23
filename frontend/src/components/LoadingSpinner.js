import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  LinearProgress,
  Chip,
} from '@mui/material';
import { GraphicEq, Psychology, CloudQueue } from '@mui/icons-material';

const LoadingSpinner = ({ message = "Processing audio..." }) => {
  const [progress, setProgress] = React.useState(0);

  React.useEffect(() => {
    const timer = setInterval(() => {
      setProgress((oldProgress) => {
        if (oldProgress === 100) {
          return 0;
        }
        const diff = Math.random() * 10;
        return Math.min(oldProgress + diff, 100);
      });
    }, 500);

    return () => {
      clearInterval(timer);
    };
  }, []);

  const processingSteps = [
    { icon: <CloudQueue />, label: "Loading Audio" },
    { icon: <GraphicEq />, label: "Extracting Features" },
    { icon: <Psychology />, label: "AI Analysis" },
  ];

  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <Card sx={{ maxWidth: 500, mx: 'auto' }}>
        <CardContent sx={{ p: 4, textAlign: 'center' }}>
          <Box sx={{ mb: 3 }}>
            <CircularProgress 
              size={60} 
              thickness={4}
              sx={{ 
                color: 'primary.main',
                animation: 'spin 2s linear infinite',
                '@keyframes spin': {
                  '0%': { transform: 'rotate(0deg)' },
                  '100%': { transform: 'rotate(360deg)' }
                }
              }}
            />
          </Box>

          <Typography variant="h5" gutterBottom fontWeight={600}>
            {message}
          </Typography>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Analyzing your audio with advanced AI models
          </Typography>

          <Box sx={{ mb: 3 }}>
            <LinearProgress 
              variant="determinate" 
              value={progress}
              sx={{ 
                height: 8, 
                borderRadius: 4,
                backgroundColor: 'grey.700',
                '& .MuiLinearProgress-bar': {
                  borderRadius: 4,
                }
              }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              {Math.round(progress)}% Complete
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap' }}>
            {processingSteps.map((step, index) => (
              <Chip 
                key={index}
                icon={step.icon}
                label={step.label}
                variant="outlined"
                color="primary"
                sx={{ 
                  '& .MuiChip-icon': { fontSize: 18 },
                  fontWeight: 500
                }}
              />
            ))}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default LoadingSpinner;
