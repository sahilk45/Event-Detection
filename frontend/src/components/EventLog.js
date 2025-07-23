import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  IconButton,
  Collapse,
  Badge,
  Divider,
} from '@mui/material';
import {
  History,
  Warning,
  Error as ErrorIcon,
  Info,
  TrendingUp,
  ExpandMore,
  Delete,
} from '@mui/icons-material';

const EventLog = () => {
  const [events, setEvents] = useState([]);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    // Load events from localStorage
    const loadEvents = () => {
      const eventLog = JSON.parse(localStorage.getItem('eventLog') || '[]');
      setEvents(eventLog.reverse()); // Show newest first
    };

    loadEvents();

    // Listen for storage changes
    const handleStorageChange = () => loadEvents();
    window.addEventListener('storage', handleStorageChange);

    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const getPriorityIcon = (priority) => {
    const icons = {
      'critical': <ErrorIcon color="error" />,
      'high': <Warning color="warning" />,
      'medium': <Info color="info" />,
      'low': <TrendingUp color="success" />,
    };
    return icons[priority] || <Info />;
  };

  const getPriorityColor = (priority) => {
    const colors = {
      'critical': 'error',
      'high': 'warning',
      'medium': 'info',
      'low': 'success',
    };
    return colors[priority] || 'default';
  };

  const clearLog = () => {
    localStorage.removeItem('eventLog');
    setEvents([]);
  };

  const alertCount = events.filter(event => 
    ['critical', 'high'].includes(event.priority)
  ).length;

  return (
    <Card>
      <CardContent sx={{ p: 0 }}>
        <Box 
          sx={{ 
            p: 2, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            cursor: 'pointer'
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Badge badgeContent={alertCount} color="error">
              <History color="primary" />
            </Badge>
            <Typography variant="h6" fontWeight={600}>
              Event Log
            </Typography>
            <Chip 
              label={`${events.length} events`}
              size="small"
              color="primary"
              variant="outlined"
            />
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {events.length > 0 && (
              <IconButton 
                size="small" 
                onClick={(e) => {
                  e.stopPropagation();
                  clearLog();
                }}
                color="error"
              >
                <Delete />
              </IconButton>
            )}
            <ExpandMore 
              sx={{ 
                transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.3s'
              }}
            />
          </Box>
        </Box>

        <Collapse in={expanded}>
          <Divider />
          {events.length === 0 ? (
            <Box sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                No events logged yet. Upload audio files to see detection history.
              </Typography>
            </Box>
          ) : (
            <List sx={{ maxHeight: 300, overflow: 'auto' }}>
              {events.slice(0, 10).map((event, index) => (
                <Box key={event.timestamp + index}>
                  <ListItem 
                    sx={{ 
                      py: 1,
                      '&:hover': { bgcolor: 'action.hover' },
                      opacity: 0,
                      animation: `fadeIn 0.5s ease-in-out ${index * 0.1}s forwards`,
                      '@keyframes fadeIn': {
                        from: { opacity: 0, transform: 'translateX(-20px)' },
                        to: { opacity: 1, transform: 'translateX(0)' }
                      }
                    }}
                  >
                    <ListItemIcon sx={{ minWidth: 40 }}>
                      {getPriorityIcon(event.priority)}
                    </ListItemIcon>
                    
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Typography variant="body2" fontWeight={600}>
                            {event.event}
                          </Typography>
                          <Chip 
                            label={event.priority.toUpperCase()}
                            size="small"
                            color={getPriorityColor(event.priority)}
                            sx={{ fontSize: '0.7rem', height: 20 }}
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            {`${(event.confidence * 100).toFixed(1)}% confidence • ${new Date(event.timestamp).toLocaleTimeString()}`}
                          </Typography>
                          <br />
                          <Typography variant="caption" color="text.secondary">
                            {event.filename}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                  {index < events.length - 1 && <Divider />}
                </Box>
              ))}
            </List>
          )}
        </Collapse>
      </CardContent>
    </Card>
  );
};

export default EventLog;
