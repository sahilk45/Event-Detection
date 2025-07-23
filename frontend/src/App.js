import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  CssBaseline,
  Container,
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  AppBar,
  Toolbar,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
} from '@mui/material';
import {
  GraphicEq,
  Info,
  ExpandMore,
  CloudUpload,
  Mic,
} from '@mui/icons-material';

import AudioUploader from './components/AudioUploader';
import ResultDisplay from './components/ResultDisplay';
import LoadingSpinner from './components/LoadingSpinner';
import EventLog from './components/EventLog';
import LiveRecording from './components/LiveRecording';
import { audioAPI } from './services/api';
import { theme } from './theme/theme';

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
}

function App() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [fileResult, setFileResult] = useState(null);
  const [liveResult, setLiveResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    // Load model info
    const loadModelInfo = async () => {
      try {
        const modelData = await audioAPI.getModelInfo();
        setModelInfo(modelData);
      } catch (err) {
        console.error('Error loading model info:', err);
      }
    };

    loadModelInfo();
  }, []);

  const handleFileSelect = async (file) => {
    setIsProcessing(true);
    setError(null);
    setFileResult(null);

    try {
      const predictionResult = await audioAPI.predictAudio(file);
      setFileResult(predictionResult);

      // Log high-confidence events
      if (predictionResult?.alert_info?.should_alert) {
        const eventLog = JSON.parse(localStorage.getItem('eventLog') || '[]');
        eventLog.push({
          event: predictionResult.alert_info.event_name,
          confidence: predictionResult.alert_info.confidence,
          priority: predictionResult.alert_info.priority,
          timestamp: predictionResult.alert_info.timestamp,
          filename: file.name,
          source: 'file_upload'
        });
        
        if (eventLog.length > 50) {
          eventLog.splice(0, eventLog.length - 50);
        }
        
        localStorage.setItem('eventLog', JSON.stringify(eventLog));
        window.dispatchEvent(new Event('storage'));
      }

    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleLivePrediction = (prediction) => {
    // Update live result display with live prediction
    setLiveResult(prediction);
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    // Clear errors when switching tabs, but keep results
    setError(null);
  };

  // Get current result based on active tab
  const getCurrentResult = () => {
    if (tabValue === 0) return fileResult;
    if (tabValue === 1) return liveResult;
    return null;
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* Header */}
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: 2,
                  background: 'linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.9) 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  mr: 2,
                  boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                }}
              >
                <GraphicEq sx={{ fontSize: 24, color: '#1976d2' }} />
              </Box>
              
              <Box>
                <Typography variant="h6" component="div" sx={{ fontWeight: 800, letterSpacing: '-0.5px' }}>
                  AudioAI
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.8, fontSize: '0.7rem', lineHeight: 1 }}>
                  Detection System
                </Typography>
              </Box>
            </Box>
            
            <Box sx={{ flexGrow: 1 }} />
            
            <Chip 
              label="LIVE"
              size="small"
              sx={{ 
                bgcolor: 'success.main',
                color: 'white',
                fontWeight: 600,
                fontSize: '0.75rem',
                animation: 'pulse 2s infinite',
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.7 }
                }
              }}
            />
          </Toolbar>
        </AppBar>

        {/* Hero Section */}
        <Box sx={{ py: 4, textAlign: 'center', bgcolor: 'primary.main', color: 'black' }}>
          <Container maxWidth="md">
            <Typography variant="h2" component="h1" gutterBottom fontWeight={800}>
              AI-Powered Audio Analysis
            </Typography>
            <Typography variant="h6" sx={{ opacity: 0.8, mb: 2 }}>
              Real-time acoustic event detection with intelligent alerts
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
              <Chip label="Real-time" />
              <Chip label="Instant-Analysis" />
              <Chip label="Smart Alerts" />
            </Box>
          </Container>
        </Box>

        {/* Main Content */}
        <Container maxWidth="xl" sx={{ py: 4 }}>
          <Grid container spacing={4}>
            {/* Main Column */}
            <Grid item xs={12} lg={8}>
              {/* Tab Navigation */}
              <Card sx={{ mb: 3 }}>
                <Tabs 
                  value={tabValue} 
                  onChange={handleTabChange}
                  variant="fullWidth"
                  sx={{ borderBottom: 1, borderColor: 'divider' }}
                >
                  <Tab 
                    icon={<CloudUpload />} 
                    label="File Upload" 
                    id="tab-0"
                    aria-controls="tabpanel-0"
                  />
                  <Tab 
                    icon={<Mic />} 
                    label="Live Recording" 
                    id="tab-1"
                    aria-controls="tabpanel-1"
                  />
                </Tabs>
              </Card>

              {/* Tab Content */}
              <TabPanel value={tabValue} index={0}>
                <AudioUploader 
                  onFileSelect={handleFileSelect}
                  isProcessing={isProcessing}
                />

                {isProcessing && (
                  <Box sx={{ mt: 3 }}>
                    <LoadingSpinner />
                  </Box>
                )}

                {(fileResult || error) && (
                  <Box sx={{ mt: 3 }}>
                    <ResultDisplay result={fileResult} error={error} />
                  </Box>
                )}
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <LiveRecording onPrediction={handleLivePrediction} />

                {liveResult && (
                  <Box sx={{ mt: 3 }}>
                    <ResultDisplay result={liveResult} error={null} />
                  </Box>
                )}
              </TabPanel>
            </Grid>

            {/* Sidebar */}
            <Grid item xs={12} lg={4}>
              <EventLog />
              
              {/* System Status */}
              <Card sx={{ mt: 3 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <Info color="primary" />
                    <Typography variant="h6" fontWeight={600}>
                      System Status
                    </Typography>
                  </Box>
                  
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Model:</Typography>
                      <Chip 
                        label={modelInfo?.model_loaded ? 'Online' : 'Offline'}
                        color={modelInfo?.model_loaded ? 'success' : 'error'}
                        size="small"
                      />
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Alerts:</Typography>
                      <Chip 
                        label={Notification.permission === 'granted' ? 'Active' : 'Disabled'}
                        color={Notification.permission === 'granted' ? 'success' : 'warning'}
                        size="small"
                      />
                    </Box>

                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Mode:</Typography>
                      <Chip 
                        label={tabValue === 0 ? 'File Upload' : 'Live Recording'}
                        color="primary"
                        size="small"
                      />
                    </Box>
                  </Box>
                </CardContent>
              </Card>

              {/* Model Info */}
              {modelInfo && !modelInfo.error && (
                <Card sx={{ mt: 3 }}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Typography variant="h6" fontWeight={600}>
                        Model Information
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            TYPE: {modelInfo.model_type}
                          </Typography>
                        </Box>
                        
                        {modelInfo.metadata?.category_mapping && (
                          <Box>
                            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                              CATEGORIES
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                              {Object.values(modelInfo.metadata.category_mapping).map((category) => (
                                <Chip 
                                  key={category}
                                  label={category}
                                  size="small"
                                  variant="outlined"
                                />
                              ))}
                            </Box>
                          </Box>
                        )}
                      </Box>
                    </AccordionDetails>
                  </Accordion>
                </Card>
              )}
            </Grid>
          </Grid>
        </Container>

        {/* Footer */}
        <Box sx={{ mt: 4, py: 3, bgcolor: 'background.paper', textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            AISOC 2025 • Acoustic Event Detection System by TechTonic
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
