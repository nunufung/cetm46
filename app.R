## app.R ##
library(shiny)
library(shinydashboard)
library(DT)
library(ggplot2)
library(caret)
library(PerformanceAnalytics)
library(evtree)
library(mvtnorm)
library(shinyjs)
library(markdown)
library(ggmosaic)


## Loading the dataset
df <- read.csv("https://github.com/nunufung/cetm46/raw/master/online_shoppers_intention.csv")
df2 <- read.csv("https://github.com/nunufung/cetm46/raw/master/online_shoppers_intention.csv")


ui <- dashboardPage(
  dashboardHeader(title = "Shopper Intensions"),
  ## Sidebar content
  dashboardSidebar(
    sidebarMenu(
      menuItem("Introduction", tabName = "intro", icon = icon("tags")),
      menuItem("Visualization: Rev vs others", tabName = "revenue", icon = icon("money-check-alt")),
      menuItem("Visualization: Cross table", tabName = "cross", icon = icon("money-check")),
      menuItem("Shopper Dataset", tabName = "data", icon = icon("globe")),
      menuItem("Confusion Matrix", tabName = "cm", icon = icon("user-astronaut")),
      menuItem("Conclusion", tabName = "conclu", icon = icon("flask"))
    )
  ),
  dashboardBody(
    
    tabItems(
      
      
      # Second Tab content
      tabItem(tabName = "revenue", 
              
              fluidRow(
                box(width=6, plotOutput("bar1")),
                box(width=6, plotOutput("bar2")),
              ),
              fluidRow(
                box(width=6, plotOutput("bar3")),
                box(width=6, plotOutput("bar4"))
              ),
              fluidRow(
                box(width=6, plotOutput("bar5")),
                box(width=6, plotOutput("bar6"))
              ),
              fluidRow(
                box(width=6, plotOutput("bar7")),
                box(width=6, plotOutput("bar8"))
              ),
              fluidRow(
                box(width=6, plotOutput("bar9")),
                box(width=6, plotOutput("bar10"))
              )
      ),
      # Third tab content
      tabItem(tabName = "cross",
              fluidRow(
                box(width=6, plotOutput("bar11")),
                box(width=6, plotOutput("bar12")),
              ),
              fluidRow(
                box(width=6, plotOutput("bar13")),
                box(width=6, plotOutput("bar14"))
              ),
              fluidRow(
                box(width=6, plotOutput("bar15")),
                box(width=6, plotOutput("bar16"))
              ),
              fluidRow(
                box(width=6, plotOutput("bar17")),
                
              )
      ),
      # Fourth tab content
      tabItem(tabName = "data",
              fluidRow(
                box(width=3, checkboxGroupInput("show_vars", "columns in listing to show:",
                                                names(df2), selected = names(df2))),
                box(width=9, dataTableOutput("table1")))),
      
      # Fifth tab content ( to be confirmed)
      tabItem(tabName = "cm",
              fluidRow(
                
                column(12,
                       includeHTML("cm.html")
                  ))),
      #First Tab content
      tabItem(tabName = "intro",
              fluidRow(
                
                column(12,
                       includeHTML("intro.html")
                ))),
      
      #Fifth Tab content
      tabItem(tabName = "conclu",
              fluidRow(
                
                column(12,
                       includeHTML("conclu.html"),
                       
                )))
    ))
)


server <- function(input, output) {
  set.seed(122)
  histdata <- rnorm(500)
  
  
  output$plot1 <- renderPlot({
    data <- histdata[seq_len(input$slider)]
    hist(data)
  })
  
  output$bar1 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = Administrative) +
      geom_bar() +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar2 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = Administrative_Duration) +
      geom_histogram(bins = 50) +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar3 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = Informational) +
      geom_bar() +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar4 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = Informational_Duration) +
      geom_histogram(bins = 50) +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar5 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = ProductRelated) +
      geom_bar() +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar6 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = ProductRelated_Duration) +
      geom_histogram(bins = 100) +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar7 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = BounceRates) +
      geom_histogram(bins = 100) +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar8 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = ExitRates) +
      geom_histogram(bins = 100) +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar9 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = PageValues) +
      geom_histogram(bins = 50) +
      facet_grid(Revenue ~ .,
                 scales = "free_y")
  })
  output$bar10 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = SpecialDay) +
      geom_bar() +
      facet_grid(Revenue ~ .,
                 scales = "free_y") +
      scale_x_continuous(breaks = seq(0, 1, 0.1))
  })
  
  # Cross Table in second tap
  
  # default theme for ggplot2
  theme_set(theme_gray())
  
  # setting default parameters for mosaic plots
  mosaic_theme = theme(axis.text.x = element_text(angle = 90,
                                                  hjust = 1,
                                                  vjust = 0.5),
                       axis.text.y = element_blank(),
                       axis.ticks.y = element_blank())
  
  output$bar11 <- renderPlot({
    df %>% 
      ggplot() +
      aes(x = Month, Revenue = ..count../nrow(df), fill = Revenue) +
      geom_bar() +
      ylab("relative frequency")
    
    month_table <- table(df$Month, df$Revenue)
    month_tab <- as.data.frame(prop.table(month_table, 2))
    colnames(month_tab) <-  c("Month", "Revenue", "perc")
    
    ggplot(data = month_tab, aes(x = Month, y = perc, fill = Revenue)) + 
      geom_bar(stat = 'identity', position = 'dodge', alpha = 2/3) + 
      xlab("Month")+
      ylab("Percent")
  })
  
  output$bar12 <- renderPlot({
    df %>% 
      ggplot() +
      geom_mosaic(aes(x = product(Revenue, OperatingSystems), fill = Revenue)) +
      mosaic_theme +
      xlab("OS Types") +
      ylab(NULL)
  })
  
  output$bar13 <- renderPlot({
    df %>% 
      ggplot() +
      geom_mosaic(aes(x = product(Revenue, Browser), fill = Revenue)) +
      mosaic_theme +
      xlab("Broswer Types") +
      ylab(NULL)
  })
  
  output$bar14 <- renderPlot({
    df %>% 
      ggplot() +
      geom_mosaic(aes(x = product(Revenue, Region), fill = Revenue)) +
      mosaic_theme +
      xlab("Regions") +
      ylab(NULL)
  })
  
  output$bar15 <- renderPlot({
    df %>% 
      ggplot() +
      geom_mosaic(aes(x = product(Revenue, TrafficType), fill = Revenue)) +
      mosaic_theme +
      xlab("Traffic Type") +
      ylab(NULL)
  })
  
  output$bar16 <- renderPlot({
    df %>% 
      ggplot() +
      geom_mosaic(aes(x = product(Revenue, VisitorType), fill = Revenue)) +
      mosaic_theme +
      xlab("Visitor Type") +
      ylab(NULL)
  })
  
  
  output$bar17 <- renderPlot({
    df %>% 
      ggplot() +
      geom_mosaic(aes(x = product(Revenue, Weekend), fill = Revenue)) +
      mosaic_theme +
      xlab("Weekend") +
      ylab(NULL)
  })
  
  output$table1 <- DT::renderDataTable({DT::datatable(df2[, input$show_vars, drop = FALSE],
                                                      options = list (
                                                        scrollX = TRUE,
                                                        class = 'cell-border stripe')
  )
  })
  
  
}



shinyApp(ui, server)