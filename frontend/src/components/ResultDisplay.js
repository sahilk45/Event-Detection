import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  AlertTitle,
  Grid,
  Divider,
  Badge,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Stack,
  Fade,
} from '@mui/material';
import {
  TrendingUp,
  Warning,
  Error as ErrorIcon,
  Info,
  NotificationsActive,
  GraphicEq,
  Assessment,
  ExpandMore,
  Schedule,
  Description,
} from '@mui/icons-material';

const ResultDisplay = ({ result, error }) => {
  // Move useEffect to the top
  React.useEffect(() => {
    if (result?.alert_info?.should_alert && result.alert_info.priority !== 'low') {
      if (Notification.permission === 'default') {
        Notification.requestPermission();
      }
      
      if (Notification.permission === 'granted') {
        const notification = new Notification(`🚨 ${result.alert_info.event_name.toUpperCase()}`, {
          body: result.alert_info.message,
          icon: '/favicon.ico',
          badge: '/favicon.ico',
          requireInteraction: result.alert_info.priority === 'critical',
          silent: false
        });

        if (result.alert_info.priority !== 'critical') {
          setTimeout(() => notification.close(), 10000);
        }
      }
    }
  }, [result?.alert_info]);

  if (error) {
    return (
      <Fade in={true}>
        <Alert 
          severity="error" 
          icon={<ErrorIcon fontSize="inherit" />}
          sx={{ 
            borderRadius: 2,
            '& .MuiAlert-message': { width: '100%' }
          }}
        >
          <AlertTitle>Analysis Error</AlertTitle>
          {error}
        </Alert>
      </Fade>
    );
  }

  if (!result) return null;

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getPriorityDetails = (priority) => {
    const details = {
      'critical': { color: 'error', icon: <ErrorIcon />, label: 'CRITICAL' },
      'high': { color: 'warning', icon: <Warning />, label: 'HIGH' },
      'medium': { color: 'info', icon: <Info />, label: 'MEDIUM' },
      'low': { color: 'success', icon: <TrendingUp />, label: 'LOW' },
      'none': { color: 'default', icon: <Info />, label: 'INFO' }
    };
    return details[priority] || details.none;
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const priorityDetails = getPriorityDetails(result?.alert_info?.priority);

  return (
    <Box sx={{ 
      opacity: 0,
      animation: 'fadeInUp 0.6s ease-out forwards',
      '@keyframes fadeInUp': {
        from: { opacity: 0, transform: 'translateY(30px)' },
        to: { opacity: 1, transform: 'translateY(0)' }
      }
    }}>
      <Stack spacing={3}>
        {/* Alert Banner */}
        {result?.alert_info?.should_alert && (
          <Alert 
            severity={priorityDetails.color}
            icon={<NotificationsActive fontSize="inherit" />}
            sx={{
              '& .MuiAlert-message': { width: '100%' },
              border: (theme) => `2px solid ${theme.palette[priorityDetails.color].main}`,
              animation: result.alert_info.priority === 'critical' ? 'pulse 2s infinite' : 'none',
              '@keyframes pulse': {
                '0%': { boxShadow: '0 0 0 0 rgba(244, 67, 54, 0.7)' },
                '70%': { boxShadow: '0 0 0 10px rgba(244, 67, 54, 0)' },
                '100%': { boxShadow: '0 0 0 0 rgba(244, 67, 54, 0)' }
              }
            }}
          >
            <AlertTitle>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip 
                  label={priorityDetails.label}
                  color={priorityDetails.color}
                  size="small"
                  sx={{ fontWeight: 700 }}
                />
                <Typography variant="h6" component="span">
                  PRIORITY ALERT
                </Typography>
              </Box>
            </AlertTitle>
            <Typography variant="body1" sx={{ mt: 1, fontWeight: 500 }}>
              {result.alert_info.message}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic' }}>
              {result.alert_info.recommended_action}
            </Typography>
          </Alert>
        )}

        {/* Main Detection Result */}
        <Card sx={{ '&:hover': { transform: 'translateY(-2px)' }, transition: 'transform 0.2s' }}>
          <CardContent sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
              <GraphicEq color="primary" sx={{ fontSize: 32 }} />
              <Typography variant="h4" component="h2" fontWeight={700}>
                Detection Result
              </Typography>
            </Box>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'primary.contrastText', borderRadius: 2 }}>
                  <Typography variant="overline" fontWeight={600}>
                    CATEGORY
                  </Typography>
                  <Typography variant="h5" fontWeight={700}>
                    {result.predicted_category}
                  </Typography>
                </Box>
              </Grid>

              <Grid item xs={12} md={6}>
                <Box sx={{ p: 2, bgcolor: getConfidenceColor(result.confidence) + '.main', color: 'white', borderRadius: 2 }}>
                  <Typography variant="overline" fontWeight={600}>
                    CONFIDENCE
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="h5" fontWeight={700}>
                      {formatPercentage(result.confidence)}
                    </Typography>
                    {result.confidence > 0.70 && (
                      <Chip 
                        label="HIGH"
                        size="small"
                        sx={{ fontWeight: 600, bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }}
                      />
                    )}
                  </Box>
                </Box>
              </Grid>
            </Grid>

            {/* Confidence Progress Bar */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Confidence Level
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={result.confidence * 100}
                color={getConfidenceColor(result.confidence)}
                sx={{ 
                  height: 8, 
                  borderRadius: 4,
                  backgroundColor: 'grey.700'
                }}
              />
            </Box>

            {/* Subcategory */}
            {result.predicted_subcategory && (
              <Box sx={{ mt: 3 }}>
                <Divider sx={{ my: 3 }} />
                <Grid container spacing={2}>
                  <Grid item xs={12} md={8}>
                    <Box sx={{ p: 2, bgcolor: 'secondary.main', color: 'secondary.contrastText', borderRadius: 2 }}>
                      <Typography variant="overline" fontWeight={600}>
                        SUBCATEGORY
                      </Typography>
                      <Typography variant="h6" fontWeight={600}>
                        {result.predicted_subcategory}
                      </Typography>
                    </Box>
                  </Grid>
                  {result.subcategory_confidence && (
                    <Grid item xs={12} md={4}>
                      <Box sx={{ p: 2, bgcolor: 'grey.800', borderRadius: 2 }}>
                        <Typography variant="overline" color="text.secondary" fontWeight={600}>
                          CONFIDENCE
                        </Typography>
                        <Typography variant="h6" fontWeight={600}>
                          {formatPercentage(result.subcategory_confidence)}
                        </Typography>
                      </Box>
                    </Grid>
                  )}
                </Grid>
              </Box>
            )}
          </CardContent>
        </Card>

        {/* Detailed Probabilities */}
        {result.class_probabilities && (
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Assessment color="primary" />
                <Typography variant="h6" fontWeight={600}>
                  All Class Probabilities
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Stack spacing={2}>
                {Object.entries(result.class_probabilities)
                  .sort(([,a], [,b]) => b - a)
                  .map(([className, probability], index) => (
                    <Box 
                      key={className} 
                      sx={{ 
                        p: 2, 
                        bgcolor: 'background.paper', 
                        borderRadius: 2,
                        opacity: 0,
                        animation: `slideIn 0.4s ease-out ${index * 0.1}s forwards`,
                        '@keyframes slideIn': {
                          from: { opacity: 0, transform: 'translateX(-20px)' },
                          to: { opacity: 1, transform: 'translateX(0)' }
                        }
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body1" fontWeight={500}>
                          {className}
                        </Typography>
                        <Badge 
                          badgeContent={probability > 0.70 ? "HIGH" : ""}
                          color="warning"
                        >
                          <Chip 
                            label={formatPercentage(probability)}
                            color={probability > 0.70 ? 'warning' : 'default'}
                            variant={probability > 0.70 ? 'filled' : 'outlined'}
                            sx={{ fontFamily: 'monospace', fontWeight: 600 }}
                          />
                        </Badge>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={probability * 100}
                        color={probability > 0.70 ? 'warning' : 'primary'}
                        sx={{ height: 6, borderRadius: 3 }}
                      />
                    </Box>
                  ))}
              </Stack>
            </AccordionDetails>
          </Accordion>
        )}

        {/* File Information */}
        {result.filename && (
          <Card>
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Description color="primary" />
                <Typography variant="h6" fontWeight={600}>
                  File Information
                </Typography>
              </Box>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 2 }}>
                    <Typography variant="caption" color="text.secondary" fontWeight={600}>
                      FILENAME
                    </Typography>
                    <Typography variant="body1" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                      {result.filename}
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={3}>
                  <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 2 }}>
                    <Typography variant="caption" color="text.secondary" fontWeight={600}>
                      FILE SIZE
                    </Typography>
                    <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                      {(result.file_size / (1024 * 1024)).toFixed(2)} MB
                    </Typography>
                  </Box>
                </Grid>
                
                {result.alert_info?.timestamp && (
                  <Grid item xs={12} sm={3}>
                    <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 2 }}>
                      <Typography variant="caption" color="text.secondary" fontWeight={600}>
                        <Schedule sx={{ fontSize: 14, mr: 0.5 }} />
                        TIME
                      </Typography>
                      <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                        {new Date(result.alert_info.timestamp).toLocaleTimeString()}
                      </Typography>
                    </Box>
                  </Grid>
                )}
              </Grid>
            </CardContent>
          </Card>
        )}

        {/* Status Indicator */}
        <Alert 
          severity={result?.alert_info?.should_alert ? 'warning' : 'success'}
          icon={result?.alert_info?.should_alert ? <NotificationsActive /> : <TrendingUp />}
          sx={{ textAlign: 'center' }}
        >
          <Typography variant="body1" fontWeight={600}>
            {result?.alert_info?.should_alert ? 
              `🚨 Alert Active - ${result.alert_info.event_name}` : 
              '👂 System Active - Listening for events...'}
          </Typography>
        </Alert>
      </Stack>
    </Box>
  );
};

export default ResultDisplay;
